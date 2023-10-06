// ===================== HRPD TetMesh =====================
// Contains:
//      - Mesh Data for Render
//      - HRPD TetConstraint
//      - HRPD Subspace Construction
// ========================================================

#pragma once

#include <vector>

#include "Simulator/PDTypeDef.hpp"
#include "STVDSampler.hpp"
#include "igl/opengl/glfw/Viewer.h"

#ifdef PD_USE_CUDA
#include "CUDA/CUDAMatrixVectorMult.hpp"
#endif

#define BASE_FUNC_CUTOFF 0.0001
// For skinning space radial-based weights
#define USE_QUARTIC_POL true


namespace PD {

struct HRPDTetMesh {
    int m_meshID = -1;
    Eigen::MatrixXd m_colors;

    PDPositions m_restpose_positions; // [N x 3]
    PDPositions m_positions; // [N x 3]
    PDTriangles m_triangles; // [N x 3]
    PDTets m_tets; // [e x 4]

    PDPositions m_velocities;

    // Mesh relevant data
    // [N x 1]
    PDVector m_vertexMasses;
    PDScalar m_normalizationMass;

    // For Tet Constraint
    PDScalar m_minStrain;
    PDScalar m_maxStrain;
    PDScalar m_stiffness_weight;
    std::vector<PDScalar> m_normalization_weight;
    std::vector<Eigen::Triplet<PDScalar>> m_selectionMatrixTris;
    std::vector<PDMatrix> m_restEdgesInv;

    // ---- For uniform show data
    std::vector<std::vector<int>> m_adjVerts1rd, m_adjVerts2rd;
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>> m_tetsPerVertex;

#ifdef PD_USE_CUDA
    CUDABufferMapping* m_GPUBufferMapper_V = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_uv = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_normals = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_ambient_vbo = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_diffuse_vbo = nullptr;
    CUDABufferMapping* m_GPUBufferMapper_V_specular_vbo = nullptr;
#endif

public:
    HRPDTetMesh(std::string meshURL);
    
    void InitTetConstraints(PDScalar stiffness_weight, PDScalar sigmaMax, PDScalar sigmaMin);
    // S_i - [3 x N]
    PDSparseMatrixRM GetSelectionMatrix(unsigned int index);
    // ST - [N x 3e]
    PDSparseMatrix GetAssemblyMatrix(bool sqrtWeights = false, bool noWeights = false);
    // positions - [4 x 3]
    EigenMatrix3 GetP(int tInd);
    EigenMatrix3 GetP(int tInd, EigenMatrix3 edges);
    
    EigenMatrix3 GetP_Corotated(int tInd);
    EigenMatrix3 GetP_Volume(int tInd);

    PDScalar GetPDEnergy(PDPositions& positions, int tInd);
    PDScalar GetStvkEnergy(PDPositions& positions, int tInd);

public:
    void ResetPositions()
    {
        m_velocities.setZero();
        m_positions = m_restpose_positions;
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
} // namespace PD
