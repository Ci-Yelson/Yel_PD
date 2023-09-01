
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
        auto miY = m_positions.col(1).minCoeff();
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

void HRPDTetMesh::IGL_SetMesh(igl::opengl::glfw::Viewer* viewer)
{
    // spdlog::info(">>> HRPDTetMesh::IGL_SetMesh()");
    // ==================== Compute color ====================
    { 
        m_colors.resize(m_positions.rows(), 1);
        // [todo]
        Eigen::MatrixXd disp_norms = (m_positions - m_restpose_positions).cast<double>().rowwise().norm();
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
        viewer->data(m_meshID).clear();
        viewer->data(m_meshID).point_size = 1.0f;
        viewer->data(m_meshID).set_mesh(m_positions.cast<double>(), m_triangles);
        viewer->data(m_meshID).set_colors(m_colors.cast<double>());

        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - data.face_based = {}", viewer->data(m_meshID).face_based);

        auto& _data = viewer->data(m_meshID);
        _data.updateGL(_data, _data.invert_normals, _data.meshgl);
        _data.dirty = 0;
        _data.meshgl.bind_mesh();

        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - F.rows() = {}, m_colors.rows() = {}", viewer->data(m_meshID).F.rows(), m_colors.rows());

        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V = {}", viewer->data(m_meshID).meshgl.vbo_V);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_uv = {}", viewer->data(m_meshID).meshgl.vbo_V_uv);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_normals = {}", viewer->data(m_meshID).meshgl.vbo_V_normals);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_ambient = {}", viewer->data(m_meshID).meshgl.vbo_V_ambient);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_diffuse = {}", viewer->data(m_meshID).meshgl.vbo_V_diffuse);
        spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - meshgl.vbo_V_specular = {}", viewer->data(m_meshID).meshgl.vbo_V_specular);
        
        if (m_GPUBufferMapper_V) delete m_GPUBufferMapper_V;
        m_GPUBufferMapper_V = new CUDABufferMapping();
        m_GPUBufferMapper_V->initBufferMap(m_positions.size() * sizeof(float), viewer->data(m_meshID).meshgl.vbo_V);

        /*if (viewer->data(m_meshID).meshgl.V_uv_vbo.size() > 0) {
            if (m_GPUBufferMapper_V_uv) delete m_GPUBufferMapper_V_uv;
            m_GPUBufferMapper_V_uv = new CUDABufferMapping();
            m_GPUBufferMapper_V_uv->initBufferMap(viewer->data(m_meshID).meshgl.V_uv_vbo.size() * sizeof(float), viewer->data(m_meshID).meshgl.vbo_V_uv);
        }*/

        /*if (m_GPUBufferMapper_V_normals) delete m_GPUBufferMapper_V_normals;
        m_GPUBufferMapper_V_normals = new CUDABufferMapping();
        m_GPUBufferMapper_V_normals->initBufferMap(viewer->data(m_meshID).meshgl.V_normals_vbo.size() * sizeof(float), viewer->data(m_meshID).meshgl.vbo_V_normals);*/

        auto& data = viewer->data(m_meshID);
        auto& meshgl = viewer->data(m_meshID).meshgl;

        if (m_GPUBufferMapper_V_ambient_vbo) delete m_GPUBufferMapper_V_ambient_vbo;
        m_GPUBufferMapper_V_ambient_vbo = new CUDABufferMapping();
        m_GPUBufferMapper_V_ambient_vbo->initBufferMap(meshgl.V_ambient_vbo.size() * sizeof(float), viewer->data(m_meshID).meshgl.vbo_V_ambient);

        if (m_GPUBufferMapper_V_diffuse_vbo) delete m_GPUBufferMapper_V_diffuse_vbo;
        m_GPUBufferMapper_V_diffuse_vbo = new CUDABufferMapping();
        m_GPUBufferMapper_V_diffuse_vbo->initBufferMap(meshgl.V_diffuse_vbo.size() * sizeof(float), viewer->data(m_meshID).meshgl.vbo_V_diffuse);

        if (m_GPUBufferMapper_V_specular_vbo) delete m_GPUBufferMapper_V_specular_vbo;
        m_GPUBufferMapper_V_specular_vbo = new CUDABufferMapping();
        m_GPUBufferMapper_V_specular_vbo->initBufferMap(meshgl.V_specular_vbo.size() * sizeof(float), viewer->data(m_meshID).meshgl.vbo_V_specular);
        
        g_InteractState.isBufferMapping = true;
    }
    if (g_InteractState.isBufferMapping) {
        auto& data = viewer->data(m_meshID);
        auto& meshgl = viewer->data(m_meshID).meshgl;
        // Input:
        //   X  #V by dim quantity
        // Output:
        //   X_vbo  #F*3 by dim scattering per corner
        const auto per_corner = [&](
                                    const Eigen::MatrixXd& X,
                                    igl::opengl::MeshGL::RowMatrixXf& X_vbo) {
            X_vbo.resize(data.F.rows() * 3, X.cols());
            for (unsigned i = 0; i < data.F.rows(); ++i)
                for (unsigned j = 0; j < 3; ++j)
                    X_vbo.row(i * 3 + j) = X.row(data.F(i, j)).cast<float>();
        };

        viewer->data(m_meshID).set_mesh(m_positions.cast<double>(), m_triangles);
        //viewer->data(m_meshID).dirty = 0;
        viewer->data(m_meshID).set_colors(m_colors.cast<double>());
        meshgl.V_vbo = data.V.cast<float>();
        //Eigen::Matrix<float, -1, 3, Eigen::RowMajor> V_uv_vbo = viewer->data(m_meshID).V_uv.cast<float>();
        //Eigen::Matrix<float, -1, 3, Eigen::RowMajor> V_normals_vbo = viewer->data(m_meshID).V_normals.cast<float>();
        //per_corner(data.V_material_ambient, meshgl.V_ambient_vbo);
        //per_corner(data.V_material_diffuse, meshgl.V_diffuse_vbo);
        //per_corner(data.V_material_specular, meshgl.V_specular_vbo);

        m_GPUBufferMapper_V->bufferMap(meshgl.V_vbo.data(), meshgl.V_vbo.size());
        //m_GPUBufferMapper_V_uv->bufferMap(V_uv_vbo.data(), V_uv_vbo.size());
        //m_GPUBufferMapper_V_normals->bufferMap(V_normals_vbo.data(), V_normals_vbo.size());
        //m_GPUBufferMapper_V_ambient_vbo->bufferMap(meshgl.V_ambient_vbo.data(), meshgl.V_ambient_vbo.size());
        //m_GPUBufferMapper_V_diffuse_vbo->bufferMap(meshgl.V_diffuse_vbo.data(), meshgl.V_diffuse_vbo.size());
        //m_GPUBufferMapper_V_specular_vbo->bufferMap(meshgl.V_specular_vbo.data(), meshgl.V_specular_vbo.size());
        //viewer->data(m_meshID).dirty = igl::opengl::MeshGL::DIRTY_TEXTURE;
        //viewer->data(m_meshID).dirty = igl::opengl::MeshGL::DIRTY_TEXTURE;
        viewer->data(m_meshID).dirty = igl::opengl::MeshGL::DIRTY_DIFFUSE | igl::opengl::MeshGL::DIRTY_SPECULAR | igl::opengl::MeshGL::DIRTY_AMBIENT;
        //viewer->data(m_meshID).dirty = 0;
    }
    else {
        viewer->data(m_meshID).set_mesh(m_positions.cast<double>(), m_triangles);
        viewer->data(m_meshID).set_colors(m_colors.cast<double>());
    }
#else
    if (m_meshID == -1) {
        m_meshID = viewer->append_mesh(true);
        viewer->data(m_meshID).clear(); // clear is expensive for large models ?
        viewer->data(m_meshID).point_size = 1.0f;
    }
    viewer->data(m_meshID).set_mesh(m_positions.cast<double>(), m_triangles);
    viewer->data(m_meshID).set_colors(m_colors.cast<double>());
#endif
    // spdlog::info(">>> HRPDTetMesh::IGL_SetMesh() - After");
}

} // namespace PD