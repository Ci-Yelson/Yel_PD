#pragma once

#include <string>
#include <vector>

#include "Simulator/PDTypeDef.hpp"
#include "igl/opengl/glfw/Viewer.h"

namespace PD {

struct QNPDTetMesh {
    int m_meshID{ -1 };
    Eigen::MatrixXd m_colors;

    PDPositions m_restpose_positions; // 3xm
    PDPositions m_positions; // 3xm
    PDTriangles m_triangles; // 3xm
    bool isTet = false;
    PDTets m_tets; // 4xtn

    unsigned int m_vertices_number; // m
    unsigned int m_system_dimension; // 3m
    // unsigned int m_expanded_system_dimension; // 6s
    // unsigned int m_expanded_system_dimension_1d; // 2s

    PDScalar m_total_mass{ 1 };

    // vertices positions/previous positions/mass
    PDVector m_restpose_positions_vec; // 1x3m
    PDVector m_current_positions_vec; // 1x3m
    PDVector m_current_velocities_vec; // 1x3m
    PDVector m_previous_positions_vec; // 1x3m
    PDVector m_previous_velocities_vec; // 1x3m
    PDSparseMatrix m_mass_matrix; // 3mx3m
    PDSparseMatrix m_inv_mass_matrix; // 3mx3m

    PDSparseMatrix m_mass_matrix_1d; // mxm
    PDSparseMatrix m_inv_mass_matrix_1d; // mxm

    // ---- For uniform show data
    std::vector<std::vector<int>> m_adjVerts1rd, m_adjVerts2rd;
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>> m_tetsPerVertex;

public:
    QNPDTetMesh(std::string meshURL);

    void ResetPositions()
    {
        m_current_velocities_vec.setZero();
        m_current_positions_vec = m_restpose_positions_vec;
        m_previous_positions_vec = m_restpose_positions_vec;
        m_previous_velocities_vec = m_current_velocities_vec;
        m_positions = m_restpose_positions;
    }
    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer, Eigen::MatrixXd colorMapData);
};

}