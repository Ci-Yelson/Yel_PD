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

    // For TetStrain Constraint
    PDScalar m_minStrain;
    PDScalar m_maxStrain;
    PDScalar m_stiffness_weight;
    std::vector<PDScalar> m_normalization_weight;
    std::vector<Eigen::Triplet<PDScalar>> m_selectionMatrixTris;
    std::vector<PDMatrix> m_restEdgesInv;

public:
    HRPDTetMesh(std::string meshURL);
    
    void InitTetConstraints(PDScalar stiffness_weight, PDScalar sigmaMax, PDScalar sigmaMin);

    // S_i - [3 x N]
    PDSparseMatrixRM GetSelectionMatrix(unsigned int index);
    // ST - [N x 3e]
    PDSparseMatrix GetAssemblyMatrix(bool sqrtWeights = false, bool noWeights = false);
    // positions - [4 x 3]
    EigenMatrix3 GetP(int tInd);
    PDScalar GetPDEnergy(PDPositions& positions, int tInd);
    PDScalar GetStvkEnergy(PDPositions& positions, int tInd);

public:
    void ResetPositions()
    {
        m_velocities.setZero();
        m_positions = m_restpose_positions;
    }
    void UpdatePosAndVel(PDPositions& pos, PDPositions& vel)
    {
        m_positions = pos;
        m_velocities = vel;
    }
    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer);
};
} // namespace PD
