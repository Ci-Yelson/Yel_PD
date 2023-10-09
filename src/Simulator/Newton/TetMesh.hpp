// ===================== Classic TetMesh =====================
// Contains:
//      - Mesh Data for Render
//      - TetConstraint
//      - Gradient and Hessian compuation
// ===========================================================

#pragma once

#include "Simulator/PDTypeDef.hpp"
#include "igl/opengl/glfw/Viewer.h"
#include <vector>

#ifdef PD_USE_CUDA
#include "CUDA/CUDAMatrixVectorMult.hpp"
#endif

namespace PD {

namespace CL {

enum MATERIAL_TYPE {
    MATERIAL_TYPE_COROT,
    MATERIAL_TYPE_StVK,
    MATERIAL_TYPE_NEOHOOKEAN,
};

struct TetMesh {
    PDPositions m_restpose_positions; // [N x 3]
    PDPositions m_positions; // [N x 3]
    PDTriangles m_triangles; // [N x 3]
    PDTets m_tets; // [e x 4]

    PDPositions m_velocities;

    int m_system_dim; // 3N

    // [N x 1]
    PDVector m_vertexMasses;
    PDScalar m_normalizationMass;
    Eigen::SparseMatrix<PDScalar, Eigen::ColMajor> m_mass_matrix; //[3N x 3N]
    Eigen::SparseMatrix<PDScalar, Eigen::ColMajor> m_mass_matrix_inv; //[3N x 3N]

    // For Tet Constraint
    MATERIAL_TYPE m_material_type = MATERIAL_TYPE::MATERIAL_TYPE_NEOHOOKEAN;
    PDScalar m_mu;
    PDScalar m_lambda;
    // -- For neohookean inverted part
    const PDScalar m_neohookean_clamp_value = 0.1;

    Eigen::Matrix<PDScalar, -1, 1> m_positions_vec; // flattened m_positions: [x0 y0 z0 x1 ...]
    Eigen::Matrix<PDScalar, -1, 1> m_velocities_vec; // flattened m_velocities: [x0 y0 z0 x1 ...]

    std::vector<Eigen::Matrix<PDScalar, 3, 3>> m_DmInvs; // Dm^{-1}
    std::vector<PDScalar> m_Ws; // 1.0 / 6.0 * std::abs(DmInv.determinant());

    PDScalar m_E = 0; // Energy
    Eigen::Matrix<PDScalar, -1, 1> m_Es; // Element-wise Energy
    Eigen::Matrix<PDScalar, -1, 1> m_Grad; // Gradient
    Eigen::SparseMatrix<PDScalar, Eigen::ColMajor> m_H; // Hessian

public:
    // For libigl render
    int m_meshID = -1;
    Eigen::MatrixXd m_colors;

#ifdef PD_USE_CUDA
    CUDABufferMapping* m_GPUBufferMapper_V = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_uv = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_normals = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_ambient_vbo = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_diffuse_vbo = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_specular_vbo = nullptr;
#endif

public:
    TetMesh(std::string meshURL);

    void InitTetConstraints(PDScalar mu, PDScalar lambda);

    // TODO
    void EvalEnergy(Eigen::Matrix<PDScalar, -1, 3>& pos);
    void EvalGradient(Eigen::Matrix<PDScalar, -1, 3>& pos);
    void EvalHessian(Eigen::Matrix<PDScalar, -1, 3>& pos, bool definitness_fix = true);

public:
    // Math utils
    void reduced_to_full(Eigen::Matrix<PDScalar, -1, 3>& reduced, Eigen::Matrix<PDScalar, -1, 1>& full);
    void full_to_reduced(Eigen::Matrix<PDScalar, -1, 3>& reduced, Eigen::Matrix<PDScalar, -1, 1>& full);

    Eigen::Matrix<PDScalar, 3, 3> Get_deformation_gradient(int tInd, Eigen::Matrix<PDScalar, -1, 3>& pos);
    PDScalar Get_element_energy(int tInd, Eigen::Matrix<PDScalar, -1, 3>& pos);
    Eigen::Matrix<PDScalar, 9, 12> Get_vec_dF_dx(int tInd, Eigen::Matrix<PDScalar, -1, 3>& pos);
    Eigen::Matrix<PDScalar, 9, 1> Get_vec_dPhi_dF(int tInd, Eigen::Matrix<PDScalar, -1, 3>& pos);
    Eigen::Matrix<PDScalar, 9, 9> Get_vec_d2Phi_dF2(int tInd, Eigen::Matrix<PDScalar, -1, 3>& pos);

    /// Matrix Projection onto Positive Semi-Definite Cone
    template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> project_to_psd(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& A);

public:
    void ResetPositions()
    {
        m_velocities.setZero();
        m_positions = m_restpose_positions;
        m_velocities_vec.setZero();
        reduced_to_full(m_positions, m_positions_vec);
#ifdef PD_USE_CUDA
        for (int d = 0; d < 3; d++) {
            Eigen::Matrix<float, -1, 3, Eigen::RowMajor> V_vbo = m_positions.cast<float>();
            m_GPUBufferMapper_V->bufferMap(V_vbo.data(), V_vbo.size());
        }
#endif
    }
    void UpdatePosAndVel(PDPositions& pos, PDPositions& vel)
    {
        m_positions = pos;
        m_velocities = vel;
    }
    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer, Eigen::MatrixXd colorMapData);
};

}

}