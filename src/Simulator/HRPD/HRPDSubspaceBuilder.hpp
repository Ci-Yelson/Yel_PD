#pragma once

#include "Simulator/HRPD/HRPDTetMesh.hpp"
#include "UI/InteractState.hpp"
#include "Simulator/PDTypeDef.hpp"
#include <memory>

#ifdef PD_USE_CUDA
#include "Simulator/HRPD/CUDA/CUDAMatrixVectorMult.hpp"
#endif

#define USE_PCA true
/* If this is turned on, the basis from the skinning
   construction is created without any weights, the interpolation will
   also be done without the weights, and only after the subspace rhs
   vector has been found, the weights are applied. */
#define NO_WEIGHTS_IN_CONSTRUCTION true
#define MIN_OVERSAMPLING 20

extern UI::InteractState g_InteractState;

namespace PD {

struct SubspaceBuilder {
    std::shared_ptr<HRPDTetMesh> m_mesh;

    // For Mesh Sample
    STVDSampler m_sampler;

    // [N x N]
    PDSparseMatrix m_massMatrixForPos;
    PDVector m_massMatrixDiagForPos;
    // [3e x 3e]
    PDSparseMatrix m_massMatrixForProj;
    PDVector m_massMatrixDiagForProj;
    PDSparseMatrix m_weightMatrixForProj;
    PDVector m_weightMatrixDiagForProj;

    std::vector<std::vector<std::pair<unsigned int, unsigned int>>> m_tetsPerVertex;

    // Subspace intepolation matrices
    PDMatrix m_U; // For Position - [N x 4k]
    PDMatrix m_UT;
    PDMatrix m_V; // For Projection - [3e x (ks + 1)]
    PDSparseMatrix m_J; // For resolution-independent - [3kt x 3e]

    PDSparseMatrix m_U_sp;
    PDSparseMatrix m_UT_sp;
    PDSparseMatrix m_V_sp;

    // Assembly Matrix - [N x 3e]
    PDSparseMatrix m_ST;

    // Samples
    std::vector<unsigned int> m_posSamples;
    std::vector<unsigned int> m_projSamples;
    std::vector<unsigned int> m_constraintVertexSamples;
    std::vector<unsigned int> m_sampledConstraintInds;

    // For online compuation
    PDMatrix ms_VTJT;
    PDMatrix ms_UTSTWiV;

    // Solver
    PDDenseSolver m_fittingSolver;

    // -- For subspace: Project fullspace to subspace
    // ---- Solve UTMU * x_{sub} = UTM * x_{full}
    // ---- Question: Why not solve UTU * x_{sub} = UT * x_{full} ?
    PDMatrix m_UTMU;
    PDSparseSolver m_subspaceSolverSparse;

    // -- For used subspace vertices
    std::vector<unsigned int> m_usedVertices;
    PDPositions m_positionsUsedVs;
    PDPositions m_velocitiesUsedVsRepulsion;
    PDPositions m_velocitiesUsedVs;
    // ---- U'T [U_{used}.transpose()] [4k x N']
    PDMatrix m_U_used;
    PDMatrix m_UT_used;
    PDSparseMatrix m_U_used_sp;
    PDSparseMatrix m_UT_used_sp;
    // ---- Solve (UT_{used} * U_{used}) * x_{sub} = (UT_{used}) * x_{used}
    // ---- Get q_{sub} from q_{used} when q_{used} is changed, i.e. collision handling.
    // ---- But why not directly solve (UT * M * U) * x_{sub} = (UT * M) * q_{full} ?
    // PDDenseLLTSolver m_usedSubspaceSolverDense;
    PDSparseSolver m_usedSubspaceSolverSparse;

public:
    void Init(std::shared_ptr<HRPDTetMesh> mesh)
    {
        m_mesh = mesh;
        assert(m_mesh->m_tets.rows() != 0);
        {
            m_sampler.Init(m_mesh->m_restpose_positions, m_mesh->m_triangles, m_mesh->m_tets);
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

    PDMatrix SnapshotPCA(PDMatrix& Y, PDVector& masses);
    // For skinning space
    PDMatrix GetRadialBaseFunctions(std::vector<unsigned int>& samples, bool partitionOfOne, double r, double eps = -1., int numSmallSamples = -1, double smallSamplesRadius = 1.);
    // [N x (4k + 1)]
    PDMatrix CreateSkinningSpace(PDPositions& restPositions, PDMatrix& weights, unsigned int linesPerConstraint = 1);

public:
    void CreateSubspacePosition();
    void CreateSubspaceProjection();
    void CreateProjectionInterpolationMatrix();

    void InitProjection();
    void InitInterpolation();

    void ProjectFullspaceToSubspaceForPos(PDPositions& sub, PDPositions& full);
    void ProjectUsedFullspaceToSubspaceForPos(PDPositions& sub, PDPositions& usedfull);
    void InterpolateSubspaceToFullspafceForPos(PDPositions& posFull, PDPositions& posSub, bool usedVerticesOnly);
};

}