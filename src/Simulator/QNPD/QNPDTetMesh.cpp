#include "QNPDTetMesh.hpp"
#include "Simulator/PDTypeDef.hpp"
#include "UI/InteractState.hpp"
#include "Util/MeshIO.hpp"
#include "Util/StoreData.hpp"

#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/util/Constants.h>
#include <igl/copyleft/tetgen/cdt.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/remove_duplicate_vertices.h>
#include <spdlog/spdlog.h>

#include "igl/jet.h"

#include <fstream>
#include <iostream>

extern UI::InteractState g_InteractState;

namespace PD {

QNPDTetMesh::QNPDTetMesh(std::string meshURL)
{
    { // Load mesh file [Support .stl .obj]
        spdlog::info("Loading mesh from {}", meshURL);
        std::string fileSuffix = meshURL.substr(meshURL.find_last_of(".") + 1);
        if (fileSuffix == "stl") {
            std::ifstream input(meshURL, std::ios::in | std::ios::binary);
            if (!input) {
                spdlog::error("Failed to open {}", meshURL);
                exit(1);
            }
            Eigen::MatrixXd V, N;
            Eigen::MatrixXi F;
            bool success = igl::readSTL(input, V, F, N);
            input.close();
            if (!success) {
                spdlog::error("Failed to read {}", meshURL);
                exit(1);
            }
            // remove duplicate vertices
            Eigen::MatrixXi SVI, SVJ;
            Eigen::MatrixXd SVd;
            igl::remove_duplicate_vertices(V, 0, SVd, SVI, SVJ);
            std::for_each(F.data(), F.data() + F.size(),
                [&SVJ](int& f) { f = SVJ(f); });
            m_positions = SVd.cast<PDScalar>();
            m_triangles = F.cast<PDIndex>();
            // info
            std::cout << "Original vertices: " << V.rows() << std::endl;
            std::cout << "Duplicate-free vertices: " << m_positions.rows() << std::endl;
        }
        else if (fileSuffix == "obj") {
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            if (!igl::readOBJ(meshURL, V, F)) {
                spdlog::error("Failed to read {}", meshURL);
                exit(1);
            }
            m_positions = V.cast<PDScalar>();
            m_triangles = F.cast<PDIndex>();
        }
        else if (fileSuffix == "msh") {
            Eigen::MatrixXd TV;
            Eigen::MatrixXi TT, TF;
            if (!Util::readTetMesh(meshURL, TV, TT, TF)) {
                spdlog::error("Failed to read {}", meshURL);
                exit(1);
            }
            m_positions = TV.cast<PDScalar>();
            m_tets = TT.cast<PDIndex>();
            m_triangles = TF.cast<PDIndex>();
            isTet = true;
        }
        m_restpose_positions = m_positions;

        if (isTet) {
            spdlog::info("Mesh loaded: {} vertices, {} faces, {} tets", m_positions.rows(), m_triangles.rows(), m_tets.rows());
        }
        else {
            spdlog::info("Mesh loaded: {} vertices, {} faces", m_positions.rows(), m_triangles.rows());
        }
    }

    {
        // TODO: UNSTABLE?
        // Scale Mesh -> 1 x 1 x 1 box.
        // spdlog::info("Scale Mesh -> 1 x 1 x 1 box.");
        // double len = std::max({ m_positions.col(0).maxCoeff() - m_positions.col(0).minCoeff(),
        //     m_positions.col(1).maxCoeff() - m_positions.col(1).minCoeff(),
        //     m_positions.col(2).maxCoeff() - m_positions.col(2).minCoeff() });
        // double factor = 1.0 / len;
        // m_positions *= factor;

        // Move Mesh - Make sure Y coordinate > 0
        auto miY = m_positions.col(1).minCoeff() - g_InteractState.dHat;
        if (miY < 0.0) {
            spdlog::info("Move Mesh - Make sure Y coordinate > 0.");
            for (int v = 0; v < m_positions.rows(); v++) {
                m_positions.row(v).y() += -miY;
            }
        }
        Util::storeData(m_positions, m_triangles, "./debug/mesh.obj", true);
    }

    if (!isTet) { // Tetrahedralize
        std::string args = "p";
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
        m_restpose_positions = m_positions;
        isTet = true;
        spdlog::info("TetrahedralizeMesh successful! {} vertices, {} faces, {} tets", m_positions.rows(), m_triangles.rows(), m_tets.rows());
    }

    { // Mesh params init
        m_vertices_number = m_positions.rows();
        m_system_dimension = m_vertices_number * 3;

        // Assign initial position, velocity and mass to all the vertices.
        m_restpose_positions_vec.resize(m_system_dimension);
        m_current_positions_vec.resize(m_system_dimension);
        m_current_velocities_vec.resize(m_system_dimension);
        m_mass_matrix.resize(m_system_dimension, m_system_dimension);
        m_inv_mass_matrix.resize(m_system_dimension, m_system_dimension);
        m_mass_matrix_1d.resize(m_vertices_number, m_vertices_number);
        m_inv_mass_matrix_1d.resize(m_vertices_number, m_vertices_number);

        // Assign initial position to all the vertices.
        for (size_t i = 0; i < m_vertices_number; i++) {
            for (int j = 0; j < 3; j++) {
                m_restpose_positions_vec[3 * i + j] = m_restpose_positions(i, j);
            }
        }
        // Eigen::Matrix<PDScalar, -1, 3, Eigen::RowMajor> m_TVrm = m_positions;
        // m_restpose_positions_vec = Eigen::Map<Eigen::Matrix<PDScalar, -1, 1, Eigen::RowMajor>>(m_positions.data(), m_system_dimension, 1); // this is wrong !!!
        // m_restpose_positions_vec = Eigen::Map<Eigen::Matrix<PDScalar, -1, 1>>(m_positions.data(), m_system_dimension, 1); // this is wrong !!!
        // Util::storeData(m_positions, "./debug/m_positions");
        // Util::storeData(m_restpose_positions_vec, "./debug/m_restpose_positions_vec");
        // spdlog::info(">>>After QNPDTetMesh::init() - ok 1");

        // Assign initial velocity to zero
        m_current_velocities_vec.setZero();
        m_current_positions_vec = m_restpose_positions_vec;
        m_previous_positions_vec = m_restpose_positions_vec;
        m_previous_velocities_vec = m_current_velocities_vec;

        // Assign mass matrix and an equally sized identity matrix
        std::vector<PDSparseMatrixTriplet> m_triplets;
        std::vector<PDSparseMatrixTriplet> m_inv_triplets;

        PDScalar unit_mass = m_total_mass / m_system_dimension;
        PDScalar inv_unit_mass = 1.0 / unit_mass;
        for (int i = 0; i < m_system_dimension; i++) {
            m_triplets.push_back(PDSparseMatrixTriplet(i, i, unit_mass));
            m_inv_triplets.push_back(PDSparseMatrixTriplet(i, i, inv_unit_mass));
        }
        m_mass_matrix.setFromTriplets(m_triplets.begin(), m_triplets.end());
        m_inv_mass_matrix.setFromTriplets(m_inv_triplets.begin(), m_inv_triplets.end());
        // spdlog::info(">>>After QNPDTetMesh::init() - ok 2");

        m_triplets.clear();
        m_inv_triplets.clear();
        for (int i = 0; i < m_vertices_number; i++) {
            m_triplets.push_back(PDSparseMatrixTriplet(i, i, unit_mass));
            m_inv_triplets.push_back(PDSparseMatrixTriplet(i, i, inv_unit_mass));
        }
        m_mass_matrix_1d.setFromTriplets(m_triplets.begin(), m_triplets.end());
        m_inv_mass_matrix_1d.setFromTriplets(m_inv_triplets.begin(), m_inv_triplets.end());

        // spdlog::info(">>>After QNPDTetMesh::init() - ok 3");
        spdlog::info(">>>After QNPDTetMesh::init()");
    }
}

void QNPDTetMesh::IGL_SetMesh(igl::opengl::glfw::Viewer* viewer)
{
    if (m_meshID == -1) {
        m_meshID = viewer->append_mesh(true);
    }
    viewer->data(m_meshID).clear();
    viewer->data(m_meshID).point_size = 1.0f;
    viewer->data(m_meshID).set_mesh(m_positions.cast<double>(), m_triangles);
    { // Compute color
        m_colors.resize(m_positions.rows(), m_positions.cols());
        // [todo]
        Eigen::MatrixXd disp_norms = (m_positions - m_restpose_positions).cast<double>().rowwise().norm();
        disp_norms /= disp_norms.maxCoeff();

        igl::jet(disp_norms, false, m_colors);
    }
    viewer->data(m_meshID).set_colors(m_colors.cast<double>());

    if (g_InteractState.draggingState.isDragging && (viewer->core().is_animating || g_InteractState.isSingleStep)) {
        auto vert = g_InteractState.draggingState.vertex;
        // spdlog::info("Pre draw - g_InteractState.draggingState points: {0} {1} {2}", pos.row(vert).x(), pos.row(vert).y(), pos.row(vert).z());
        viewer->data(m_meshID).point_size = 20.f;
        auto C = Eigen::RowVector3d(0, 1, 0);
        viewer->data(m_meshID).set_points(m_positions.cast<double>().row(vert), C);
    }
}

}