#include "Simulator/PDTypeDef.hpp"
#include "UI/InteractState.hpp"
#include <Simulator/HRPD/HRPDSubspaceBuilder.hpp>

#include "Util/Profiler.hpp"
#include "Util/StoreData.hpp"
#include "Viewer/PDViewer.hpp"
#include "spdlog/spdlog.h"

#ifdef PD_USE_CUDA
#include "CUDA/CUDAMatrixOP.hpp"
#endif

extern UI::InteractState g_InteractState;
extern Util::Profiler g_FrameProfiler;
extern Util::Profiler g_StepProfiler;
extern Util::Profiler g_PreComputeProfiler;

namespace PD {

void SubspaceBuilder::Init(std::shared_ptr<HRPDTetMesh> mesh)
{
    m_mesh = mesh;
    assert(m_mesh->m_tets.rows() != 0);
    {
        m_sampler.Init(m_mesh->m_restpose_positions, m_mesh->m_triangles, m_mesh->m_tets);
        m_velocitiesRepulsion.resize(m_mesh->m_velocities.rows(), m_mesh->m_velocities.cols());
    }
    { // computeMassMatrix() and computeWeightMatrix()
        int _N = m_mesh->m_positions.rows();
        int _Nt = m_mesh->m_tets.rows();
        // -- For position space
        // [N x N]
        m_massMatrixForPos.resize(_N, _N);
        std::vector<PDSparseMatrixTriplet> massTrisPos(_N);
        PD_PARALLEL_FOR
        for (int v = 0; v < _N; v++) {
            massTrisPos[v] = { v, v, m_mesh->m_vertexMasses(v, 0) };
        }
        m_massMatrixForPos.setFromTriplets(massTrisPos.begin(), massTrisPos.end());
        m_massMatrixForPos.makeCompressed();

        // -- For projection space
        m_massMatrixForProj.resize(_Nt * 3, _Nt * 3);
        m_massMatrixDiagForProj.resize(_Nt * 3);
        m_weightMatrixForProj.resize(_Nt * 3, _Nt * 3);
        m_weightMatrixDiagForProj.resize(_Nt * 3);
        std::vector<PDSparseMatrixTriplet> massTrisProj(_Nt * 3);
        std::vector<PDSparseMatrixTriplet> weightTris(_Nt * 3);
        PD_PARALLEL_FOR
        for (int tid = 0; tid < _Nt; ++tid) {
            double mass = 0;
            for (int d = 0; d < 4; d++) {
                mass += m_mesh->m_vertexMasses(m_mesh->m_tets(tid, d));
            }
            for (int d = 0; d < 3; d++) {
                massTrisProj[tid * 3 + d] = { tid * 3 + d, tid * 3 + d, mass };
                m_massMatrixDiagForProj(tid * 3 + d) = mass;
            }
            for (int d = 0; d < 3; d++) {
                if (NO_WEIGHTS_IN_CONSTRUCTION) {
                    // Currently true
                    weightTris[tid * 3 + d] = { tid * 3 + d, tid * 3 + d, m_mesh->m_normalization_weight[tid] };
                    m_weightMatrixDiagForProj(tid * 3 + d) = m_mesh->m_normalization_weight[tid];
                }
                else {
                    weightTris[tid * 3 + d] = { tid * 3 + d, tid * 3 + d, 1.0 };
                    m_weightMatrixDiagForProj(tid * 3 + d) = 1.0;
                }
            }
        }
        m_massMatrixForProj.setFromTriplets(massTrisProj.begin(), massTrisProj.end());
        m_massMatrixForProj.makeCompressed();
        m_weightMatrixForProj.setFromTriplets(weightTris.begin(), weightTris.end());
        m_weightMatrixForProj.makeCompressed();

        spdlog::info("Mass Matrix and Weight Matrix - Done.");
    }

    {
        // build tetsPerVertex
        // [4 x e]
        m_tetsPerVertex.clear();
        m_tetsPerVertex.resize(m_mesh->m_positions.rows());
        for (int tet = 0; tet < m_mesh->m_tets.rows(); tet++) {
            for (int v = 0; v < 4; v++) {
                unsigned int vInd = m_mesh->m_tets(tet, v);
                m_tetsPerVertex[vInd].push_back({ tet, v });
            }
        }
    }
}

PDMatrix SubspaceBuilder::SnapshotPCA(PDMatrix& Y, PDVector& masses)
{
    PROFILE_PREC("SnapshotPCA");

    spdlog::info("Performing snapshot PCA...");
    int size = g_InteractState.hrpdParams.numberSamplesForVertexProjSubspace;
    // [3e x (size + 1)] - size = 2ks
    PDMatrix basis(Y.rows(), size + 1);
    // Remove column wise mean
    {
        PROFILE_PREC("SUB_COL_MEAN");
        Eigen::VectorXd col_means = Y.colwise().mean();
        Y.rowwise() -= col_means.transpose();
    }
    // We solve the SVD by solving an eigenvalue problem on the smaller matrix A = Y^T M Y
    // and converting the eigenvectors to that problem back to v = Y u
    spdlog::info("    Computing Y^T M Y...");
    PDMatrix A;
    A.resize(Y.cols(), Y.cols());
    // TODO: When use cuda for 177k model case, numerical problem sometimes occured?
#ifdef PD_USE_CUDA_PRE
    CUDAMatrixVectorMultiplier* yMulti;
#endif
    {
        PROFILE_PREC("COMPUTE_A");
#ifdef PD_USE_CUDA_PRE
        spdlog::info("		Uploading Y to GPU...");
        yMulti = new CUDAMatrixVectorMultiplier(Y, masses);
        spdlog::info("		Computing Y^T (M Y) on GPU...");
        spdlog::info("A size = ({}, {})", A.rows(), A.cols());
        spdlog::info("M size = ({}, {})", masses.rows(), masses.cols());
        for (unsigned int i = 0; i < Y.cols(); i++) {
            PDScalar one = 1;
            yMulti->mult(Y.data() + (i * Y.rows()), A.data() + (i * A.rows()), one, true);
        }
#else
        {
            PDMatrix Y_T = Y.transpose();
            PD_PARALLEL_FOR
            for (int j = 0; j < Y.cols(); j++) {
                Y_T.col(j) *= masses(j);
            }
            A = Y_T * Y;
            // `A = Y.transpose() * m_massMatrix * Y` will cause error!
        }
#endif
    }

    Eigen::Matrix<std::complex<PDScalar>, -1, -1> eVecsC;
    Eigen::Matrix<std::complex<PDScalar>, -1, 1> eValsC;
    {
        PROFILE_PREC("COMPUTE_V");
        spdlog::info("	Computing eigenvectors v of Y^T M Y...");
        Eigen::EigenSolver<PDMatrix> eSolver;
        eSolver.compute(A, true);
        eVecsC = eSolver.eigenvectors();
        eValsC = eSolver.eigenvalues();

        // auto eVec = eSolver.eigenvectors().real();
        // auto eVal = eSolver.eigenvalues().real();
        // Util::storeData(eVec, "debug/#eVec", true);
        // Util::storeData(eVal, "debug/#eVal", true);
    }

    {
        PROFILE_PREC("EXTRACT_V");
        spdlog::info("	Extracting PCA vectors as Y v...");
        PDScalar currentEVal = -1;
        std::vector<unsigned int> eVecInds;
        for (int v = 0; v < size; v++) {
            PDScalar currentLargest = 0;
            int indOfLargest = -1;
            for (int j = 0; j < eValsC.rows(); j++) {
                if (eValsC(j).real() > currentLargest && (currentEVal < 0 || eValsC(j).real() < currentEVal)) {
                    currentLargest = eValsC(j).real();
                    indOfLargest = j;
                }
            }
            if (indOfLargest >= 0) {
                eVecInds.push_back(indOfLargest);
                currentEVal = eValsC(indOfLargest).real();
            }
            else {
                spdlog::info("Could only find {} eigenvectors with strictly positive eigenvalues during snapshot PCA ...", (v - 1));
                basis.conservativeResize(basis.rows(), std::max(1, v - 10));
                break;
            }
        }

#ifdef PD_USE_CUDA_PRE
        for (int v = 0; v < std::min((int)basis.cols(), (int)eVecInds.size()); v++) {
            PDScalar weight = (1. / std::sqrt(eValsC(eVecInds[v]).real()));
            PDVector eVec = eVecsC.col(eVecInds[v]).real();
            yMulti->mult(eVec.data(), basis.data() + (v * basis.rows()), weight, false);
        }
        delete yMulti;
#else
        {
            int sz = std::min((int)basis.cols(), (int)eVecInds.size());
            PDMatrix B(eVecsC.rows(), sz);
            PD_PARALLEL_FOR
            for (int v = 0; v < sz; v++) {
                B.col(v) = eVecsC.col(eVecInds[v]).real() * (1. / std::sqrt(eValsC(eVecInds[v]).real()));
            }
            basis.block(0, 0, Y.rows(), sz) = Y * B;
        }
#endif
    }
    if (basis.hasNaN()) spdlog::warn("Warning: basis contains NaN values!");
    // Make sure to include global translations since we removed column wise mean
    basis.col(basis.cols() - 1).setConstant(1.);

    spdlog::info("	Done.");

    return basis;
}

PDMatrix SubspaceBuilder::GetRadialBaseFunctions(std::vector<unsigned int>& samples,
    bool partitionOfOne, double r,
    double eps, int numSmallSamples,
    double smallSampleRadius)
{
    if (eps < 0) eps = std::sqrt(-std::log(BASE_FUNC_CUTOFF)) / r;
    // size = (NV * samples.size())
    int _rows = m_mesh->m_restpose_positions.rows();
    int _cols = samples.size();
    PDMatrix baseFuncs(_rows, _cols);
    baseFuncs.setZero();
    double a = 1.0 / std::pow(r, 4.0);
    double b = -2.0 * (1.0 / (r * r));

    // TIMECOST COUNT: samples.size() * (NV + NV * log(NV))
    for (int i = 0; i < _cols; i++) {
        if (numSmallSamples > 0 && i > _cols - numSmallSamples) {
            r = smallSampleRadius;
            eps = std::sqrt(-std::log(BASE_FUNC_CUTOFF)) / r;
        }

        auto curSample = samples[i];
        m_sampler.m_srcVerts.clear();
        m_sampler.m_srcVerts.push_back(curSample);
        m_sampler.computeDistances(false, STVD_MODE_GRAPH_DIJKSTRA, r);

        PD_PARALLEL_FOR
        for (int v = 0; v < _rows; v++) {
            double curDist = m_sampler.m_dists[v];
            double weight = 0;
            if (curDist < 0) {
                weight = 0;
            }
            else if (USE_QUARTIC_POL) {
                if (curDist >= r)
                    weight = 0;
                else
                    weight = a * std::pow(curDist, 4.0) + b * (curDist * curDist) + 1;
            }
            else {
                weight = std::exp(-(curDist * eps * curDist * eps));
                if (weight < BASE_FUNC_CUTOFF)
                    weight = 0;
            }
            baseFuncs(v, i) = weight;
        }
    }

    // normalization
    if (partitionOfOne) {
        for (int v = 0; v < _rows; v++) {
            double sum = baseFuncs.row(v).sum();
            if (sum < 1e-6) {
                spdlog::critical("Warning: a vertex isn't properly covered by any of the radial basis functions!");
                int col = baseFuncs.row(v).cwiseAbs().maxCoeff();
                baseFuncs(v, col) = 1.;
            }
            else {
                baseFuncs.row(v) /= sum;
            }
        }
    }

    return baseFuncs;
}

PDMatrix SubspaceBuilder::CreateSkinningSpace(PDPositions& restPositions,
    PDMatrix& weights,
    unsigned int linesPerConstraint)
{
    if (weights.hasNaN() || restPositions.hasNaN()) {
        spdlog::critical("Warning: weights or rest-state used to create skinning space have NaN values!");
    }

    int numGroups = weights.cols(); // Samples size [k]
    int numRows = restPositions.rows();
    // [N x (4k + 1)]
    PDMatrix skinningSpace(numRows, numGroups * 4 + 1);

    bool error = false;
    PD_PARALLEL_FOR
    for (int v = 0; v < numRows; v++) {
        for (int g = 0; g < numGroups; g++) {
            double curWeight = 0;
            curWeight = weights(v / linesPerConstraint, g);
            if (std::isnan(curWeight)) {
                spdlog::critical("Warning: NaN weight during skinning space construction!");
                error = true;
                break;
            }
            for (int d = 0; d < 3; d++) {
                skinningSpace(v, g * 4 + d) = curWeight * restPositions(v, d);
            }
            skinningSpace(v, g * 4 + 3) = curWeight;
        }
    }
    if (error) {
        return PDMatrix(0, 0);
    }

    skinningSpace.col(skinningSpace.cols() - 1).setConstant(1.);

    if (skinningSpace.hasNaN()) {
        spdlog::critical("Warning: skinning space has NaN values!");
    }

    return skinningSpace;
}

void SubspaceBuilder::CreateSubspacePosition()
{
    PROFILE_PREC("CREATE-POS-SUBSPACE");
    m_posSamples = m_sampler.GetSamples(g_InteractState.hrpdParams.numberSamplesForVertexPosSubspace);
    double maxDist = m_sampler.GetMaxDist(m_posSamples);
    double radius = maxDist * g_InteractState.hrpdParams.radiusMultiplierForPosSubspace;
    // [N x k]
    PDMatrix baseFunctionWeights = GetRadialBaseFunctions(m_posSamples, true, radius);
    // [N x (4k + 1)]
    m_U = CreateSkinningSpace(m_mesh->m_restpose_positions, baseFunctionWeights, 1);
    m_UT = m_U.transpose();
    m_U_sp = m_U.sparseView(0, PD_SPARSITY_CUTOFF);
    m_UT_sp = m_UT.sparseView(0, PD_SPARSITY_CUTOFF);
}

void SubspaceBuilder::CreateSubspaceProjection()
{
    PROFILE_PREC("CreateSubspaceProjection");
    m_projSamples = m_sampler.GetSamples(g_InteractState.hrpdParams.numberSamplesForVertexProjSubspace * 0.5);
    double maxDist = m_sampler.GetMaxDist(m_projSamples);
    double radius = maxDist * g_InteractState.hrpdParams.radiusMultiplierForProjSubspace;
    // [N x (0.5 * ks)]
    PDMatrix baseFunctionWeights = GetRadialBaseFunctions(m_projSamples, true, radius);
    // Rescaling weights for tets...
    PDTets& tets = m_mesh->m_tets;
    PDMatrix _projWeights(tets.rows(), baseFunctionWeights.cols());
    PD_PARALLEL_FOR
    for (int t = 0; t < tets.rows(); t++) {
        for (unsigned int c = 0; c < baseFunctionWeights.cols(); c++) {
            double curWeight = 0;
            for (unsigned d = 0; d < 4; d++) {
                curWeight += baseFunctionWeights(tets(t, d), c);
            }
            curWeight /= 4.0;
            _projWeights(t, c) = curWeight;
        }
        _projWeights.row(t) /= _projWeights.row(t).sum();
    }

    // Create assembly matrixx
    // ST - [N x 3e]
    m_ST = m_mesh->GetAssemblyMatrix(false, NO_WEIGHTS_IN_CONSTRUCTION);
    // P_{rest} = S @ Q_{rest} -> [3e x 3] = [3e x N] @ [N x 3]
    PDPositions restP = m_ST.transpose() * m_mesh->m_restpose_positions;
    // [3e x (2ks + 1)]
    PDMatrix Y = CreateSkinningSpace(restP, _projWeights, 3);
    if (!USE_PCA) {
        m_V = Y;
    }
    else {
        m_V = SnapshotPCA(Y, m_massMatrixDiagForProj);
    }
}

void SubspaceBuilder::CreateProjectionInterpolationMatrix()
{
    PROFILE_PREC("CreateProjectionInterpolationMatrix");
    // ### Extend vertex sample to constraint sample
    TICK(EXTEND);
    int numConstraintSamples = g_InteractState.hrpdParams.numberSampledConstraints;
    m_constraintVertexSamples = m_posSamples;
    { // make sure m_constraintVertexSamples size = numConstraintSamples
        if (m_constraintVertexSamples.empty()) {
            m_constraintVertexSamples = m_sampler.GetSamples(numConstraintSamples);
        }
        else if (m_constraintVertexSamples.size() < numConstraintSamples) {
            m_sampler.m_srcVerts = m_constraintVertexSamples;
            m_sampler.m_dists.setConstant(-1.);
            m_sampler.ExtendSamples(numConstraintSamples - m_constraintVertexSamples.size(), m_constraintVertexSamples);
        }
        while (m_constraintVertexSamples.size() < numConstraintSamples) {
            // Fill with rand value
            int randVal = std::rand() % static_cast<int>(m_mesh->m_positions.rows());
            auto fd = std::find(m_constraintVertexSamples.begin(), m_constraintVertexSamples.end(), randVal);
            if (fd == m_constraintVertexSamples.end()) {
                m_constraintVertexSamples.push_back(randVal);
            }
        }
    }
    // Build m_sampledConstraintInds
    m_sampledConstraintInds.clear();
    m_sampledConstraintInds.reserve(numConstraintSamples);
    for (auto& v : m_constraintVertexSamples) {
        if (m_tetsPerVertex[v].size() > 0) {
            // Just push the first tet - one vert one tet sample
            // int tet = m_tetsPerVertex[v][0].first;
            // auto fd = std::find(m_sampledConstraintInds.begin(), m_sampledConstraintInds.end(), tet);
            // if (fd == m_sampledConstraintInds.end()) {
            //     m_sampledConstraintInds.push_back(tet);
            // }
            // Or as much as possible
            for (auto tet : m_tetsPerVertex[v]) {
                auto fd = std::find(m_sampledConstraintInds.begin(), m_sampledConstraintInds.end(), tet.first);
                if (fd == m_sampledConstraintInds.end()) {
                    m_sampledConstraintInds.push_back(tet.first);
                    break;
                }
            }
        }
    }
    std::sort(m_sampledConstraintInds.begin(), m_sampledConstraintInds.end());
    spdlog::info(">>> ConstraintTetSamples size = {}", m_sampledConstraintInds.size());
    TOCK(EXTEND);

    // ### Set used vertices [?]
    {
        TICK(SETUP_USEDSPACE);
        m_usedVertices.resize(m_sampledConstraintInds.size() * 4);
        PD_PARALLEL_FOR
        for (int subTetInd = 0; subTetInd < m_sampledConstraintInds.size(); subTetInd++) {
            for (int d = 0; d < 4; d++) {
                int fullTetInd = m_sampledConstraintInds[subTetInd];
                m_usedVertices[subTetInd * 4 + d] = m_mesh->m_tets(fullTetInd, d);
            }
        }
        std::sort(m_usedVertices.begin(), m_usedVertices.end());
        m_usedVertices.erase(std::unique(m_usedVertices.begin(), m_usedVertices.end()), m_usedVertices.end());
        m_sampledTetsForUsedVs.resize(m_sampledConstraintInds.size(), 4);
        PD_PARALLEL_FOR
        for (int subTetInd = 0; subTetInd < m_sampledConstraintInds.size(); subTetInd++) {
            int fullTetInd = m_sampledConstraintInds[subTetInd];
            for (int d = 0; d < 4; d++) {
                for (int vsInd = 0; vsInd < m_usedVertices.size(); vsInd++) {
                    if (m_usedVertices[vsInd] == m_mesh->m_tets(fullTetInd, d)) {
                        m_sampledTetsForUsedVs(subTetInd, d) = vsInd;
                        break;
                    }
                }
            }
        }

        m_positionsUsedVs.setZero(m_usedVertices.size(), 3);
        m_velocitiesUsedVsRepulsion.setZero(m_usedVertices.size(), 3);
        m_velocitiesUsedVs.setZero(m_usedVertices.size(), 3);
        // Get ready for the subspace interpolation. [Interpolate subspace to fullspace]
        // A - U_{used} - [4s x 4k]
        PDMatrix A(m_usedVertices.size(), m_U.cols());
        PD_PARALLEL_FOR
        for (int i = 0; i < m_usedVertices.size(); i++) {
            A.row(i) = m_U.row(m_usedVertices[i]);
        }
        m_U_used = A;
        m_U_used_sp = m_U_used.sparseView(0, PD_SPARSITY_CUTOFF);
        m_UT_used = m_U_used.transpose();
        m_UT_used_sp = m_UT_used.sparseView(0, PD_SPARSITY_CUTOFF);

        PDMatrix lhsMat = A.transpose() * A; // [4kt x 4kt]

        // m_usedSubspaceSolverDense.compute(lhsMat);
        // if (m_usedSubspaceSolverDense.info() != Eigen::Success) {
        //     spdlog::warn("Warning: Factorization of the Dense lhs matrix for used vertex interoplation was not successful!");
        // }

        PDSparseMatrix lhsMatSparse = lhsMat.sparseView(0, PD_SPARSITY_CUTOFF);
        m_usedSubspaceSolverSparse.compute(lhsMatSparse);
        if (m_usedSubspaceSolverSparse.info() != Eigen::Success) {
            spdlog::warn("Warning: Factorization of the SPARSE lhs matrix for used vertex interoplation was not successful!");
            PDSparseMatrix eps(lhsMatSparse.rows(), lhsMatSparse.rows());
            eps.setIdentity();
            eps *= 1e-12;
            while (m_usedSubspaceSolverSparse.info() != Eigen::Success && eps.coeff(0, 0) < 1e-10) {
                spdlog::warn("Adding small diagonal entries ({})...", eps.coeff(0, 0));
                lhsMatSparse += eps;
                eps *= 2;
                m_usedSubspaceSolverSparse.compute(lhsMatSparse);
            }
        }
        TOCK(SETUP_USEDSPACE);
    }

    {
#ifdef PD_USE_CUDA
        spdlog::info(">>> GPU Updater Init - Before");
        if (m_vPosGPUUpdater != nullptr) delete m_vPosGPUUpdater;
        m_vPosGPUUpdater = new CUDASparseMatrixVectorMultiplier(m_U_sp);
        // if (m_vsPosGPUUpdater) delete m_vsPosGPUUpdater;
        // m_vsPosGPUUpdater = new CUDASparseMatrixVectorMultiplier(m_U_used_sp);

        spdlog::info(">>> GPU Updater Init - After");
#endif
    }

    // ### Create selection J matrix
    TICK(SELECTION_J);
    // 3 * kt
    std::vector<PDSparseMatrixTriplet> selectedInds(m_sampledConstraintInds.size() * 3);
    for (size_t i = 0; i < m_sampledConstraintInds.size(); i++) {
        for (size_t d = 0; d < 3; d++) {
            selectedInds.at(i * 3 + d) = PDSparseMatrixTriplet{ PD::PDIndex(i * 3 + d), PD::PDIndex(m_sampledConstraintInds[i] * 3 + d), 1.0 };
        }
    }
    // J -> [3kt x 3e]
    m_J.resize(3 * m_sampledConstraintInds.size(), 3 * m_mesh->m_tets.rows());
    m_J.setFromTriplets(selectedInds.begin(), selectedInds.end());
    if (m_J.rows() < m_V.cols() - MIN_OVERSAMPLING) {
        spdlog::warn("  Warning: not enough sampled constraints ({}) for this rhs interpolation base ({}).", m_sampledConstraintInds.size(), m_V.cols());
        spdlog::info("  Cutting off last vectors in base.");
        m_V.conservativeResize(m_V.rows(), m_J.rows() - MIN_OVERSAMPLING);
    }
    m_J.makeCompressed();
    TOCK(SELECTION_J);
    spdlog::info(">>> Selection matrix J - done.");
}

void SubspaceBuilder::InitProjection()
{
    PROFILE_PREC("InitProjection");
#ifdef PD_USE_CUDA
    CUDAMatrixUTMU(m_U, m_massMatrixForPos, m_UTMU);
#else
    m_UTMU = m_UT * m_massMatrixForPos * m_U;
#endif
    PDSparseMatrix UTMU_Sparse = m_UTMU.sparseView(0, PD_SPARSITY_CUTOFF);
    m_subspaceSolverSparse.compute(UTMU_Sparse);
    if (m_subspaceSolverSparse.info() != Eigen::Success) {
        spdlog::warn("Warning: Factorization of Subspace Solver Sparse was not successful!");
    }
}

void SubspaceBuilder::InitInterpolation()
{
    PROFILE_PREC("InitInterpolation");
    ms_VTJT = m_V.transpose() * m_J.transpose();
    PDMatrix _VTJTJV = ms_VTJT * (m_J * m_V);
    m_fittingSolver.compute(_VTJTJV);
    // 4) compute finalize matrix
    // Finalize Matrix = [4k x (ks + 1)] -> UT @ ST (@ WI) @ V
    {
        TICKC(INTERPOLATION_FINALIZE);
        TICK(SPARSE_PART_TIMECOST);
        // UT @ ST -> [4k x 3e] = [4k x N] @ [N x 3e]
        PDSparseMatrix UTST_WI = m_UT_sp * m_ST;
        PD_PARALLEL_FOR
        for (int c = 0; c < UTST_WI.cols(); c++) {
            UTST_WI.col(c) *= m_weightMatrixDiagForProj(c);
        }
        TOCK(SPARSE_PART_TIMECOST);
        TICK(DENSE_PART_TIMECOST);
        // Finalize Matrix = [4k x (ks + 1)] -> UT @ ST (@ WI) @ V
        ms_UTSTWiV.resize(m_UT_sp.rows(), m_V.cols());
#ifdef PD_USE_CUDA //[TODO]
        CUDASparseMatrixVectorMultiplier* cudaMulti = new CUDASparseMatrixVectorMultiplier(UTST_WI);
        for (int c = 0; c < m_V.cols(); c++) {
            // [4k x 1] = [4k x 3e] [3e x 1]
            // ms_UTSTWiV.col(c) = UTST_WI * m_V.col(c);
            cudaMulti->mult(m_V.data() + (c * m_V.rows()), ms_UTSTWiV.data() + (c * ms_UTSTWiV.rows()));
        }
        delete cudaMulti;
        // { // Check
        //     PD_PARALLEL_FOR
        //     PDMatrix _db(ms_UTSTWiV.rows(), ms_UTSTWiV.cols());
        //     for (int c = 0; c < m_V.cols(); c++) {
        //         // [4k x 1] = [4k x 3e] [3e x 1]
        //         _db.col(c) = UTST_WI * m_V.col(c);
        //     }
        //     spdlog::critical(">>> SubspaceBuilder::InitInterpolation - Finalize Matrix Diff = {}", (_db - ms_UTSTWiV).rowwise().norm().sum());
        //     // Util::storeData(ms_UTSTWiV, "debug/UTSTWiV_GPU", true);
        //     // Util::storeData(_db, "debug/UTSTWiV_CPU", true);
        // }
#else
        PD_PARALLEL_FOR
        for (int c = 0; c < m_V.cols(); c++) {
            // [4k x 1] = [4k x 3e] [3e x 1]
            ms_UTSTWiV.col(c) = UTST_WI * m_V.col(c);
        }
#endif
        TOCK(DENSE_PART_TIMECOST);
        TOCKC(INTERPOLATION_FINALIZE);
    }
}

void SubspaceBuilder::ProjectFullspaceToSubspaceForPos(PDPositions& sub, PDPositions& full)
{
    // Solve UTMU * x_{sub} = UTM * x_{full}
    PDSparseMatrix _UTM = m_UT_sp * m_massMatrixForPos;
    sub.resize(m_UT.rows(), 3);
    PD_PARALLEL_FOR
    for (int d = 0; d < 3; d++) {
        sub.col(d) = m_subspaceSolverSparse.solve(_UTM * full.col(d));
    }
}

void SubspaceBuilder::ProjectUsedFullspaceToSubspaceForPos(PDPositions& sub, PDPositions& usedfull)
{
    // Solve (UT_{used} * U_{used}) * x_{sub} = (UT_{used}) * x_{used}
    sub.resize(m_UT_used_sp.rows(), 3);
    PD_PARALLEL_FOR
    for (int d = 0; d < 3; d++) {
        sub.col(d) = m_usedSubspaceSolverSparse.solve(m_UT_used_sp * usedfull.col(d));
    }
}

void SubspaceBuilder::InterpolateSubspaceToFullspaceForPos(PDPositions& posFull, PDPositions& posSub, bool usedVerticesOnly)
{
    // x_{full} = U * x_{sub}
    if (usedVerticesOnly) {
        { // GPU updater is slower than CPU - Matrix size not big enough
            PROFILE_STEP("INTERPOLATION_USED");
            PD_PARALLEL_FOR
            for (int d = 0; d < 3; d++) {
                posFull.col(d) = m_U_used_sp * posSub.col(d);
            }
        }
    }
    else {
#ifdef PD_USE_CUDA
        { // Sparse Matrix GPU Implementation - Faster
            PROFILE_STEP("INTERPOLATION_FULL");
            PDScalar one = 1;
            for (int d = 0; d < 3; d++) {
                // m_vPosGPUUpdater->mult(posSub.data() + (d * posSub.rows()), nullptr, one, false, d, int(m_mesh->m_positions.rows()));
                m_vPosGPUUpdater->mult(posSub.data() + (d * posSub.rows()), posFull.data() + (d * posFull.rows()), d, int(m_mesh->m_positions.rows()));
            }
        }
        // { // DEBUG
        //     PROFILE_STEP("INTERPOLATION_FULL_DB");
        //     PDPositions db_posFull = posFull;
        //     PD_PARALLEL_FOR
        //     for (int d = 0; d < 3; d++) {
        //         db_posFull.col(d) = m_U_sp * posSub.col(d);
        //     }
        //     spdlog::critical(">>> SubspaceBuilder::InterpolateSubspaceToFullspaceForPos() - Full POS Diff = {}", (db_posFull - posFull).rowwise().norm().sum());
        // }
        // {
        //     PROFILE_STEP("INTERPOLATION_FULL");
        //     PD_PARALLEL_FOR
        //     for (int d = 0; d < 3; d++) {
        //         posFull.col(d) = m_U_sp * posSub.col(d);
        //     }
        // }
#else
        {
            PROFILE_STEP("INTERPOLATION_FULL");
            PD_PARALLEL_FOR
            for (int d = 0; d < 3; d++) {
                posFull.col(d) = m_U_sp * posSub.col(d);
            }
        }
#endif
    }
}

void SubspaceBuilder::GetUsedP()
{
    // Should update m_positionsUsedVs before.
    m_projectionsUsedVs.resize(3 * m_sampledConstraintInds.size(), 3);
    // TODO: If use m_positionsUsedVs to compute m_projectionsUsedVs, the "rotation artifact" becomes more obvious?
    PD_PARALLEL_FOR
    for (int tInd = 0; tInd < m_sampledConstraintInds.size(); tInd++) {
        // 3d edges of tet
        EigenMatrix3 edges;
        edges.col(0) = (m_positionsUsedVs.row(m_sampledTetsForUsedVs(tInd, 1)) - m_positionsUsedVs.row(m_sampledTetsForUsedVs(tInd, 0)));
        edges.col(1) = (m_positionsUsedVs.row(m_sampledTetsForUsedVs(tInd, 2)) - m_positionsUsedVs.row(m_sampledTetsForUsedVs(tInd, 0)));
        edges.col(2) = (m_positionsUsedVs.row(m_sampledTetsForUsedVs(tInd, 3)) - m_positionsUsedVs.row(m_sampledTetsForUsedVs(tInd, 0)));

        m_projectionsUsedVs.block(tInd * 3, 0, 3, 3) = m_mesh->GetP(m_sampledConstraintInds[tInd], edges);
    }
    // PD_PARALLEL_FOR
    // for (int tInd = 0; tInd < m_sampledConstraintInds.size(); tInd++) {
    //     m_projectionsUsedVs.block(tInd * 3, 0, 3, 3) = m_mesh->GetP(m_sampledConstraintInds[tInd]);
    // }
}
}