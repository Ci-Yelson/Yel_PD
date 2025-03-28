
#include "HRPDTetMesh.hpp"

#include <igl/copyleft/tetgen/cdt.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/remove_duplicate_vertices.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Simulator/PDTypeDef.hpp"
#include "UI/InteractState.hpp"
#include "Util/StoreData.hpp"
#include "Util/Timer.hpp"

#include "igl/jet.h"

extern UI::InteractState g_InteractState;

namespace PD {

HRPDTetMesh::HRPDTetMesh(std::string meshURL)
{
    // Load mesh
    spdlog::info("Loading mesh from {}", meshURL);
    std::string fileSuffix = meshURL.substr(meshURL.find_last_of(".") + 1);
    if (fileSuffix == "stl") {
        std::ifstream input(meshURL, std::ios::in | std::ios::binary);
        if (!input) {
            spdlog::error("Failed to open {}", meshURL);
            exit(1);
        }
        Eigen::MatrixXd _V, _N;
        Eigen::MatrixXi _F;
        bool success = igl::readSTL(input, _V, _F, _N);
        input.close();
        if (!success) {
            spdlog::error("Failed to read {}", meshURL);
            exit(1);
        }
        // remove duplicate vertices
        Eigen::MatrixXi _SVI, _SVJ;
        Eigen::MatrixXd _SVd;
        igl::remove_duplicate_vertices(_V, 0, _SVd, _SVI, _SVJ);
        std::for_each(_F.data(), _F.data() + _F.size(),
            [&_SVJ](int& f) { f = _SVJ(f); });
        m_positions = _SVd.cast<double>();
        m_triangles = _F.cast<int>();
        // info
        std::cout << "Original vertices: " << m_positions.rows() << std::endl;
        std::cout << "Duplicate-free vertices: " << m_positions.rows() << std::endl;
    }
    else if (fileSuffix == "obj") {
        Eigen::MatrixXd _V;
        Eigen::MatrixXi _F;
        if (!igl::readOBJ(meshURL, _V, _F)) {
            spdlog::error("Failed to read {}", meshURL);
            exit(1);
        }
        m_positions = _V.cast<double>();
        m_triangles = _F.cast<int>();
    }
    else if (fileSuffix == "off") {
        Eigen::MatrixXd _V;
        Eigen::MatrixXi _F;
        if (!igl::read_triangle_mesh(meshURL, _V, _F)) {
            spdlog::error("Failed to read {}", meshURL);
            exit(1);
        }
        m_positions = _V.cast<double>();
        m_triangles = _F.cast<int>();
    }

    // igl::read_triangle_mesh(meshURL, m_positions, m_triangles);
    spdlog::info("Mesh loaded: {} vertices, {} faces", m_positions.rows(), m_triangles.rows());

    {
        // Scale Mesh -> 1 x 1 x 1 box.
        spdlog::info("Scale Mesh -> 1 x 1 x 1 box.");
        double len = std::max({ m_positions.col(0).maxCoeff() - m_positions.col(0).minCoeff(),
            m_positions.col(1).maxCoeff() - m_positions.col(1).minCoeff(),
            m_positions.col(2).maxCoeff() - m_positions.col(2).minCoeff() });
        double factor = 1.0 / len;
        m_positions *= factor;

        // Move Mesh - Make sure Y coordinate > 0
        auto miY = m_positions.col(1).minCoeff() - g_InteractState.dHat;
        if (miY < 0.0) {
            spdlog::info("Move Mesh - Make sure Y coordinate > 0.");
            for (int v = 0; v < m_positions.rows(); v++) {
                m_positions.row(v).y() += -miY;
            }
        }
    }

    { // Tetrahedralize
        std::string args = "pY";
        spdlog::info("TetGen args: -{}", args);
        Eigen::Matrix<double, -1, 3> V, TV;
        Eigen::Matrix<int, -1, 3> F, TF;
        V = m_positions.cast<double>();
        F = m_triangles.cast<int>();
        // tetrahedralize() need use `double`
        int flag = igl::copyleft::tetgen::tetrahedralize(V, F, args, TV, m_tets, TF);
        // spdlog::info("TetGen flag: {}", flag);
        if (flag != 0) {
            spdlog::error("TetrahedralizeMesh failed!");
        }
        // invert normals
        for (int i = 0; i < TF.rows(); i++) {
            std::swap(TF(i, 1), TF(i, 2));
        }
        m_positions = TV.cast<PDScalar>();
        m_triangles = TF.cast<PDIndex>();
        spdlog::info("TetrahedralizeMesh successful! {} vertices, {} faces", m_positions.rows(), m_triangles.rows());
    }

    m_restpose_positions = m_positions;
    m_velocities.resize(m_positions.rows(), m_positions.cols());
    m_velocities.setZero();

    { // Mesh relevant data
        int _N = m_positions.rows();
        int _Nt = m_tets.rows();
        // -- For position space
        m_vertexMasses = PDVector(_N);
        m_vertexMasses.setZero();
        auto& pos = m_positions;
        auto& tet = m_tets;
        // PD_PARALLEL_FOR - will cause error because of `+=` operation
        for (int tInd = 0; tInd < _Nt; tInd++) {
            Eigen::Matrix<PDScalar, 3, 3> edges;
            edges.col(0) = pos.row(tet(tInd, 1)) - pos.row(tet(tInd, 0));
            edges.col(1) = pos.row(tet(tInd, 2)) - pos.row(tet(tInd, 0));
            edges.col(2) = pos.row(tet(tInd, 3)) - pos.row(tet(tInd, 0));
            double vol = std::abs(edges.determinant()) / 6.0;
            vol /= 4.0;

            m_vertexMasses(tet(tInd, 0), 0) += vol;
            m_vertexMasses(tet(tInd, 1), 0) += vol;
            m_vertexMasses(tet(tInd, 2), 0) += vol;
            m_vertexMasses(tet(tInd, 3), 0) += vol;
        }
        PD_PARALLEL_FOR
        for (int vInd = 0; vInd < _N; vInd++) {
            m_vertexMasses(vInd, 0) = std::max(m_vertexMasses(vInd, 0), PDScalar(PD_MIN_MASS));
        }
        double totalMass = m_vertexMasses.sum();
        spdlog::info("Average vertex mass: {}", totalMass * 1.0 / _N);
        // Normalize vertex masses to integrate to 1 for numerical reasons
        m_normalizationMass = 1.0 / totalMass;
        spdlog::info("Normalize vertex mass: {}", m_normalizationMass);
        m_vertexMasses *= m_normalizationMass * g_InteractState.hrpdParams.massPerUnitArea;
    }

    { // For uniform show data - Establish neighbourhood structure for tets or triangles
        m_adjVerts1rd.resize(m_positions.rows());
        m_adjVerts2rd.resize(m_positions.rows());
        for (int i = 0; i < m_tets.rows(); i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = j + 1; k < 4; k++) {
                    m_adjVerts1rd[m_tets(i, j)].push_back(m_tets(i, k));
                    m_adjVerts1rd[m_tets(i, k)].push_back(m_tets(i, j));
                }
            }
        }
        for (int v = 0; v < m_positions.rows(); v++) {
            for (auto nv1 : m_adjVerts1rd[v]) {
                {
                    auto fd = std::find(m_adjVerts2rd[v].begin(), m_adjVerts2rd[v].end(), nv1);
                    if (fd == m_adjVerts2rd[v].end()) {
                        m_adjVerts2rd[v].push_back(nv1);
                    }
                }
                for (auto nv2 : m_adjVerts1rd[nv1]) {
                    auto fd = std::find(m_adjVerts2rd[v].begin(), m_adjVerts2rd[v].end(), nv2);
                    if (fd == m_adjVerts2rd[v].end()) {
                        m_adjVerts2rd[v].push_back(nv2);
                    }
                }
            }
        }
        // Build tetsPerVertex
        m_tetsPerVertex.clear();
        m_tetsPerVertex.resize(m_positions.rows());
        for (int i = 0; i < m_tets.rows(); i++) {
            for (int v = 0; v < 4; v++) {
                unsigned int vInd = m_tets(i, v);
                m_tetsPerVertex[vInd].push_back({ i, v });
            }
        }
    }
}

void HRPDTetMesh::InitTetConstraints(PDScalar stiffness_weight, PDScalar sigmaMax, PDScalar sigmaMin)
{
    spdlog::info("HRPDTetMesh::initTetConstraint()");
    m_stiffness_weight = stiffness_weight;
    m_minStrain = sigmaMin;
    m_maxStrain = sigmaMax;
    // [12e]
    m_selectionMatrixTris.resize(12 * m_tets.rows());
    // [3e x 3]
    m_restEdgesInv = std::vector<PDMatrix>(m_tets.rows(), PDMatrix(3, 3));
    // [e x 1]
    m_normalization_weight = std::vector<PDScalar>(m_tets.rows());
    PD_PARALLEL_FOR
    for (int tInd = 0; tInd < m_tets.rows(); tInd++) {
        int v0 = m_tets(tInd, 0);
        int v1 = m_tets(tInd, 1);
        int v2 = m_tets(tInd, 2);
        int v3 = m_tets(tInd, 3);

        Eigen::Matrix<PDScalar, 3, 3> edges;
        edges.col(0) = m_restpose_positions.row(v1) - m_restpose_positions.row(v0);
        edges.col(1) = m_restpose_positions.row(v2) - m_restpose_positions.row(v0);
        edges.col(2) = m_restpose_positions.row(v3) - m_restpose_positions.row(v0);
        PDMatrix& restEdgesInv = m_restEdgesInv[tInd];
        restEdgesInv = edges.inverse();

        PDScalar vol = std::abs((edges).determinant()) / 6.0f;
        m_normalization_weight[tInd] = m_stiffness_weight * m_normalizationMass * vol;

        // Si * q = F.transpose()
        // The selection matrix computes the current deformation gradient w.r.t
        // the current positions (i.e. multiplication of the current edges with
        // the inverse of the original edge matrix)
        int count = 0;
        for (int r = 0; r < 3; r++) {
            m_selectionMatrixTris[tInd * 12 + (count++)] = Eigen::Triplet<PDScalar>(r, v0, -(restEdgesInv(0, r) + restEdgesInv(1, r) + restEdgesInv(2, r)));
            m_selectionMatrixTris[tInd * 12 + (count++)] = Eigen::Triplet<PDScalar>(r, v1, restEdgesInv(0, r));
            m_selectionMatrixTris[tInd * 12 + (count++)] = Eigen::Triplet<PDScalar>(r, v2, restEdgesInv(1, r));
            m_selectionMatrixTris[tInd * 12 + (count++)] = Eigen::Triplet<PDScalar>(r, v3, restEdgesInv(2, r));
        }
    }
}

// ############################### For Projective Dynamics ###############################

PDSparseMatrixRM HRPDTetMesh::GetSelectionMatrix(unsigned int index)
{
    // return [3 x N]
    PDSparseMatrixRM selM(3, m_restpose_positions.rows());
    selM.setFromTriplets(m_selectionMatrixTris.begin() + index * 12,
        m_selectionMatrixTris.begin() + (index + 1) * 12);
    selM.makeCompressed();
    return selM;
}

PDSparseMatrix HRPDTetMesh::GetAssemblyMatrix(bool sqrtWeights, bool noWeights)
{
    // ST - [N x 3e]
    PDSparseMatrix ST(m_restpose_positions.rows(), m_tets.rows() * 3);
    // [12e]
    std::vector<Eigen::Triplet<PDScalar>> tripletListAssembly(m_tets.rows() * 12);
    PD_PARALLEL_FOR
    for (int i = 0; i < m_tets.rows(); i++) {
        // [N x 3]
        double weight = 1.0;
        // if (!noWeights) {
        //     weight = m_sweight[i];
        //     if (sqrtWeights) {
        //         weight = std::sqrt(weight);
        //     }
        //     if (weight < 1e-10) {
        //         spdlog::warn("Small weight!");
        //     }
        // }
        int count = 0;
        for (int k = 0; k < 12; k++) {
            auto& tri = m_selectionMatrixTris[i * 12 + k];
            tripletListAssembly[i * 12 + (count++)] = Eigen::Triplet<PDScalar>(tri.col(), i * 3 + tri.row(), tri.value() * weight);
        }
    }
    ST.setFromTriplets(tripletListAssembly.begin(), tripletListAssembly.end());
    ST.makeCompressed();

    return ST;
}

EigenMatrix3 HRPDTetMesh::GetP(int tInd)
{
    // 3d edges of tet
    Eigen::Matrix<PDScalar, 3, 3> edges;
    edges.col(0) = (m_positions.row(m_tets(tInd, 1)) - m_positions.row(m_tets(tInd, 0)));
    edges.col(1) = (m_positions.row(m_tets(tInd, 2)) - m_positions.row(m_tets(tInd, 0)));
    edges.col(2) = (m_positions.row(m_tets(tInd, 3)) - m_positions.row(m_tets(tInd, 0)));

    // Compute the deformation gradient (current edges times inverse of original edges)
    Eigen::Matrix<PDScalar, 3, 3> F = edges * m_restEdgesInv[tInd];
    // Compute SVD
    Eigen::JacobiSVD<Eigen::Matrix<PDScalar, 3, 3>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    PD3dVector S = svd.singularValues();
    // Clamp singular values
    auto clamp = [](PDScalar x, PDScalar min, PDScalar max) {
        return std::max(min, std::min(max, x));
    };
    S(0) = clamp(S(0), m_minStrain, m_maxStrain);
    S(1) = clamp(S(1), m_minStrain, m_maxStrain);
    S(2) = clamp(S(2), m_minStrain, m_maxStrain);
    if ((svd.matrixU() * svd.matrixV()).determinant() < 0.0)
        S(2) = -S(2);

    // Compute clamped deformation gradient
    Eigen::Matrix<PDScalar, 3, 3> FStar = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

    if (FStar.hasNaN()) {
        FStar.setIdentity();
    }

    return FStar.transpose();
}

EigenMatrix3 HRPDTetMesh::GetP(int tInd, EigenMatrix3 edges)
{
    // For used vertices.
    // Compute the deformation gradient (current edges times inverse of original edges)
    Eigen::Matrix<PDScalar, 3, 3> F = edges * m_restEdgesInv[tInd];
    // Compute SVD
    Eigen::JacobiSVD<Eigen::Matrix<PDScalar, 3, 3>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    PD3dVector S = svd.singularValues();
    // Clamp singular values
    auto clamp = [](PDScalar x, PDScalar min, PDScalar max) {
        return std::max(min, std::min(max, x));
    };
    S(0) = clamp(S(0), m_minStrain, m_maxStrain);
    S(1) = clamp(S(1), m_minStrain, m_maxStrain);
    S(2) = clamp(S(2), m_minStrain, m_maxStrain);
    if ((svd.matrixU() * svd.matrixV()).determinant() < 0.0)
        S(2) = -S(2);

    // Compute clamped deformation gradient
    Eigen::Matrix<PDScalar, 3, 3> FStar = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

    if (FStar.hasNaN()) {
        FStar.setIdentity();
    }

    return FStar.transpose();
}

EigenMatrix3 HRPDTetMesh::GetP_Corotated(int tInd)
{
    // 3d edges of tet
    Eigen::Matrix<PDScalar, 3, 3> edges;
    edges.col(0) = (m_positions.row(m_tets(tInd, 1)) - m_positions.row(m_tets(tInd, 0)));
    edges.col(1) = (m_positions.row(m_tets(tInd, 2)) - m_positions.row(m_tets(tInd, 0)));
    edges.col(2) = (m_positions.row(m_tets(tInd, 3)) - m_positions.row(m_tets(tInd, 0)));

    // Compute the deformation gradient (current edges times inverse of original edges)
    Eigen::Matrix<PDScalar, 3, 3> F = edges * m_restEdgesInv[tInd];

    auto PolarDecomposition = [](const Eigen::Matrix<PDScalar, 3, 3>& F, Eigen::Matrix<PDScalar, 3, 3>& R, Eigen::Matrix<PDScalar, 3, 3>& S) {
        const Eigen::JacobiSVD<Eigen::Matrix<PDScalar, 3, 3>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix<PDScalar, 3, 3> Sig = svd.singularValues().asDiagonal();
        const Eigen::Matrix<PDScalar, 3, 3> U = svd.matrixU();
        const Eigen::Matrix<PDScalar, 3, 3> V = svd.matrixV();
        R = U * V.transpose();
        S = V * Sig * V.transpose();
    };

    Eigen::Matrix<PDScalar, 3, 3> R, S;
    PolarDecomposition(F, R, S);
    return R;
}

EigenMatrix3 HRPDTetMesh::GetP_Volume(int tInd)
{
    // 3d edges of tet
    Eigen::Matrix<PDScalar, 3, 3> edges;
    edges.col(0) = (m_positions.row(m_tets(tInd, 1)) - m_positions.row(m_tets(tInd, 0)));
    edges.col(1) = (m_positions.row(m_tets(tInd, 2)) - m_positions.row(m_tets(tInd, 0)));
    edges.col(2) = (m_positions.row(m_tets(tInd, 3)) - m_positions.row(m_tets(tInd, 0)));

    // Compute the deformation gradient (current edges times inverse of original edges)
    Eigen::Matrix<PDScalar, 3, 3> F = edges * m_restEdgesInv[tInd];
    // Compute SVD
    Eigen::JacobiSVD<Eigen::Matrix<PDScalar, 3, 3>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    PD3dVector sig = svd.singularValues();
    PDMatrix U = svd.matrixU();
    PDMatrix V = svd.matrixV();
    // F = U * sig * V.transpose();
    // Now solve the problem:
    // min \|sig - sig*\|^2
    // s.t. \Pi sig* = 1.
    // See the appendix of the 2014 PD paper for the derivation.
    // Let D = sig* - sig.
    // min \|D\|^2 s.t. \Pi (sig(i) + D(i)) = 1.

    const PDScalar eps = std::numeric_limits<PDScalar>::epsilon();
    // CheckError(sig.minCoeff() >= eps, "Singular F.");

    // Initial guess.
    Eigen::Matrix<PDScalar, 3, 1> D = sig / std::pow(sig.prod(), PDScalar(1) / 3) - sig;
    const int max_iter = 50;
    for (int i = 0; i < max_iter; ++i) {
        // Compute C(D) = \Pi (sig_i + D_i) - 1.
        const PDScalar C = (sig + D).prod() - 1;
        // Compute grad C(D).
        Eigen::Matrix<PDScalar, 3, 1> grad_C = Eigen::Matrix<PDScalar, 3, 1>::Ones();
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                if (j == k) continue;
                grad_C(j) *= (sig(k) + D(k));
            }
        }
        const Eigen::Matrix<PDScalar, 3, 1> D_next = (grad_C.dot(D) - C) / grad_C.squaredNorm() * grad_C;
        const PDScalar diff = (D_next - D).cwiseAbs().maxCoeff();
        if (diff <= eps) break;
        // Update.
        D = D_next;
    }
    return U * (sig + D).asDiagonal() * V.transpose();
}

PDScalar HRPDTetMesh::GetPDEnergy(PDPositions& positions, int tInd)
{
    // positions - [4 x 3]
    // 3d edges of tet
    Eigen::Matrix<PDScalar, 3, 3> edges;
    edges.col(0) = (positions.row(1) - positions.row(0));
    edges.col(1) = (positions.row(2) - positions.row(0));
    edges.col(2) = (positions.row(3) - positions.row(0));

    // Compute the deformation gradient (current edges times inverse of original edges)
    Eigen::Matrix<PDScalar, 3, 3> F = edges * m_restEdgesInv[tInd];
    // Compute SVD
    Eigen::JacobiSVD<Eigen::Matrix<PDScalar, 3, 3>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    PD3dVector S = svd.singularValues();
    // Clamp singular values
    auto clamp = [](PDScalar x, PDScalar min, PDScalar max) {
        return std::max(min, std::min(max, x));
    };
    S(0) = clamp(S(0), m_minStrain, m_maxStrain);
    S(1) = clamp(S(1), m_minStrain, m_maxStrain);
    S(2) = clamp(S(2), m_minStrain, m_maxStrain);
    if ((svd.matrixU() * svd.matrixV()).determinant() < 0.0)
        S(2) = -S(2);
    Eigen::Matrix<PDScalar, 3, 3> FStar = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
    if (FStar.hasNaN()) {
        FStar.setIdentity();
    }

    double scale = 1e6;
    return m_stiffness_weight * m_normalization_weight[tInd] * 0.5 * (F - FStar).norm() * scale;
    // return 0.5 * (F - FStar).norm();
}

PDScalar HRPDTetMesh::GetStvkEnergy(PDPositions& positions, int tInd)
{
    // positions - [4 x 3]
    // 3d edges of tet
    Eigen::Matrix<PDScalar, 3, 3> edges;
    edges.col(0) = (positions.row(1) - positions.row(0));
    edges.col(1) = (positions.row(2) - positions.row(0));
    edges.col(2) = (positions.row(3) - positions.row(0));

    // Compute the deformation gradient (current edges times inverse of original edges)
    Eigen::Matrix<PDScalar, 3, 3> F = edges * m_restEdgesInv[tInd];

    double scale = 1e6;
    return m_stiffness_weight * m_normalization_weight[tInd] * 0.5 * (F.transpose() * F - Eigen::Matrix<PDScalar, 3, 3>::Identity()).norm() * scale;
    // return 0.5 * (F.transpose() * F - Eigen::Matrix<PDScalar,3,3>::Identity()).norm();
}

void HRPDTetMesh::IGL_SetMesh(igl::opengl::glfw::Viewer* viewer, Eigen::MatrixXd colorMapData)
{
    // spdlog::info(">>> HRPDTetMesh::IGL_SetMesh()");
    // ==================== Compute color ====================
    {
        m_colors.resize(m_positions.rows(), 1);
        // [todo]
        // Eigen::MatrixXd disp_norms = (m_positions - m_restpose_positions).cast<double>().rowwise().norm();
        Eigen::MatrixXd disp_norms = colorMapData;
        // spdlog::info(">>> DISP_NORMS - MIN = {}, MAX = {}", disp_norms.minCoeff(), disp_norms.maxCoeff());
        disp_norms = (disp_norms.array() >= PD_CUTOFF).select(disp_norms, 0);
        // spdlog::info(">>> DISP_NORMS - MIN = {}, MAX = {}", disp_norms.minCoeff(), disp_norms.maxCoeff());
        disp_norms /= disp_norms.maxCoeff();
        igl::jet(disp_norms, false, m_colors);
        // spdlog::info(">>> M_COLORS Size = ({}, {})", m_colors.rows(), m_colors.cols());
        // spdlog::info(">>> M_COLORS - MAX = {}", m_colors.maxCoeff());
    }

    // ==================== Set mesh data ====================
#ifdef PD_USE_CUDA
    if (m_meshID == -1) {
        m_meshID = viewer->append_mesh(true);
        auto& meshData = viewer->data(viewer->mesh_index(m_meshID));
        meshData.clear();
        meshData.point_size = 1.0f;
        meshData.set_mesh(m_positions.cast<double>(), m_triangles);
        meshData.set_colors(m_colors.cast<double>());

        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - data.face_based = {}", meshData.face_based);

        auto& _data = meshData;
        _data.updateGL(_data, _data.invert_normals, _data.meshgl);
        _data.dirty = 0;
        _data.meshgl.bind_mesh();

        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - F.rows() = {}, m_colors.rows() = {}", meshData.F.rows(), m_colors.rows());

        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V = {}", meshData.meshgl.vbo_V);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_uv = {}", meshData.meshgl.vbo_V_uv);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_normals = {}", meshData.meshgl.vbo_V_normals);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_ambient = {}", meshData.meshgl.vbo_V_ambient);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_diffuse = {}", meshData.meshgl.vbo_V_diffuse);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_specular = {}", meshData.meshgl.vbo_V_specular);

        if (m_GPUBufferMapper_V) delete m_GPUBufferMapper_V;
        m_GPUBufferMapper_V = new CUDABufferMapping();
        m_GPUBufferMapper_V->initBufferMap(m_positions.size() * sizeof(float), meshData.meshgl.vbo_V);

        /*if (meshData.meshgl.V_uv_vbo.size() > 0) {
            if (m_GPUBufferMapper_V_uv) delete m_GPUBufferMapper_V_uv;
            m_GPUBufferMapper_V_uv = new CUDABufferMapping();
            m_GPUBufferMapper_V_uv->initBufferMap(meshData.meshgl.V_uv_vbo.size() * sizeof(float), meshData.meshgl.vbo_V_uv);
        }*/

        auto& data = meshData;
        auto& meshgl = meshData.meshgl;

        if (m_GPUBufferMapper_V_normals) delete m_GPUBufferMapper_V_normals;
        m_GPUBufferMapper_V_normals = new CUDABufferMapping();
        m_GPUBufferMapper_V_normals->initBufferMap(meshgl.V_normals_vbo.size() * sizeof(float), meshgl.vbo_V_normals);

        if (m_GPUBufferMapper_V_ambient_vbo) delete m_GPUBufferMapper_V_ambient_vbo;
        m_GPUBufferMapper_V_ambient_vbo = new CUDABufferMapping();
        m_GPUBufferMapper_V_ambient_vbo->initBufferMap(meshgl.V_ambient_vbo.size() * sizeof(float), meshgl.vbo_V_ambient);

        if (m_GPUBufferMapper_V_diffuse_vbo) delete m_GPUBufferMapper_V_diffuse_vbo;
        m_GPUBufferMapper_V_diffuse_vbo = new CUDABufferMapping();
        m_GPUBufferMapper_V_diffuse_vbo->initBufferMap(meshgl.V_diffuse_vbo.size() * sizeof(float), meshgl.vbo_V_diffuse);

        if (m_GPUBufferMapper_V_specular_vbo) delete m_GPUBufferMapper_V_specular_vbo;
        m_GPUBufferMapper_V_specular_vbo = new CUDABufferMapping();
        m_GPUBufferMapper_V_specular_vbo->initBufferMap(meshgl.V_specular_vbo.size() * sizeof(float), meshgl.vbo_V_specular);

        g_InteractState.isBufferMapping = true;
    }
    if (g_InteractState.isBufferMapping) {
        auto& data = viewer->data(viewer->mesh_index(m_meshID));
        auto& meshgl = data.meshgl;

        { // Position update
            // data.set_mesh(m_positions.cast<double>(), m_triangles);
            meshgl.V_vbo = m_positions.cast<float>();
            m_GPUBufferMapper_V->bufferMap(meshgl.V_vbo.data(), meshgl.V_vbo.size());
            // meshgl.V_normals_vbo = data.V_normals.cast<float>();
            // m_GPUBufferMapper_V_normals->bufferMap(meshgl.V_normals_vbo.data(), meshgl.V_normals_vbo.size());
        }

        {
            // UV -- not used
            // m_GPUBufferMapper_V_uv->bufferMap(V_uv_vbo.data(), V_uv_vbo.size());
        }

        { // Colors update
            // meshData.set_colors(m_colors.cast<double>());
            // using MatrixXd = Eigen::MatrixXd;
            // // Ambient color should be darker color
            // const auto ambient = [](const MatrixXd& C) -> MatrixXd {
            //     MatrixXd T = 0.1 * C;
            //     T.col(3) = C.col(3);
            //     return T;
            // };
            // // Specular color should be a less saturated and darker color: dampened
            // // highlights
            // const auto specular = [](const MatrixXd& C) -> MatrixXd {
            //     const double grey = 0.3;
            //     MatrixXd T = grey + 0.1 * (C.array() - grey);
            //     T.col(3) = C.col(3);
            //     return T;
            // };
            // for (unsigned i = 0; i < data.V_material_diffuse.rows(); ++i) {
            //     // m_colors.cols() == 3
            //     data.V_material_diffuse.row(i) << m_colors.row(i), 1;
            // }
            // data.V_material_ambient = ambient(data.V_material_diffuse);
            // data.V_material_specular = specular(data.V_material_diffuse);
            // // Per-vertex material settings
            // meshgl.V_ambient_vbo = data.V_material_ambient.cast<float>();
            // meshgl.V_diffuse_vbo = data.V_material_diffuse.cast<float>();
            // meshgl.V_specular_vbo = data.V_material_specular.cast<float>();

            // [177K case] SET_COLORS: 0.0047059 s - TODO: Use cuda kernel to update?
            // TICKC(SET_COLORS);
            PD_PARALLEL_FOR
            for (size_t i = 0; i < meshgl.V_diffuse_vbo.rows(); i++) {
                Eigen::RowVector3f rC = m_colors.row(i).cast<float>();
                meshgl.V_diffuse_vbo.row(i) << rC, 1;
                meshgl.V_ambient_vbo.row(i) << (0.1 * rC), 1;
                meshgl.V_specular_vbo.row(i) << (0.3 + 0.1 * (rC.array() - 0.3)), 1;
            }

            m_GPUBufferMapper_V_ambient_vbo->bufferMap(meshgl.V_ambient_vbo.data(), meshgl.V_ambient_vbo.size());
            m_GPUBufferMapper_V_diffuse_vbo->bufferMap(meshgl.V_diffuse_vbo.data(), meshgl.V_diffuse_vbo.size());
            m_GPUBufferMapper_V_specular_vbo->bufferMap(meshgl.V_specular_vbo.data(), meshgl.V_specular_vbo.size());

            // TOCKC(SET_COLORS);
        }

        //  data.dirty = igl::opengl::MeshGL::DIRTY_DIFFUSE | igl::opengl::MeshGL::DIRTY_SPECULAR | igl::opengl::MeshGL::DIRTY_AMBIENT;
        data.dirty = 0;
    }
    else {
        auto& data = viewer->data(viewer->mesh_index(m_meshID));
        data.set_mesh(m_positions.cast<double>(), m_triangles);
        data.set_colors(m_colors.cast<double>());
    }
#else
    if (m_meshID == -1) {
        m_meshID = viewer->append_mesh(true);
        viewer->data(viewer->mesh_index(m_meshID)).clear(); // clear is expensive for large models ?
        viewer->data(viewer->mesh_index(m_meshID)).point_size = 1.0f;
    }
    auto& meshData = viewer->data(viewer->mesh_index(m_meshID));
    meshData.set_mesh(m_positions.cast<double>(), m_triangles);
    meshData.set_colors(m_colors.cast<double>());
#endif
    // spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - After");
}

} // namespace PD