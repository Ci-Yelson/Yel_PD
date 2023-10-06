#include "TetMesh.hpp"

#include "Eigen/SVD"

#include <igl/copyleft/tetgen/cdt.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/remove_duplicate_vertices.h>
#include <spdlog/spdlog.h>

#include "UI/InteractState.hpp"

extern UI::InteractState g_InteractState;

namespace PD {

namespace CL {

TetMesh::TetMesh(std::string meshURL)
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

    {
        // Mesh relevant data
        // int _N = m_positions.rows();
        // int _Nt = m_tets.rows();
        // // -- For position space
        // m_vertexMasses = PDVector(_N);
        // m_vertexMasses.setZero();
        // auto& pos = m_positions;
        // auto& tet = m_tets;
        // // PD_PARALLEL_FOR - will cause error because of `+=` operation
        // for (int tInd = 0; tInd < _Nt; tInd++) {
        //     Eigen::Matrix<PDScalar, 3, 3> edges;
        //     edges.col(0) = pos.row(tet(tInd, 1)) - pos.row(tet(tInd, 0));
        //     edges.col(1) = pos.row(tet(tInd, 2)) - pos.row(tet(tInd, 0));
        //     edges.col(2) = pos.row(tet(tInd, 3)) - pos.row(tet(tInd, 0));
        //     double vol = std::abs(edges.determinant()) / 6.0;
        //     vol /= 4.0;

        //     m_vertexMasses(tet(tInd, 0), 0) += vol;
        //     m_vertexMasses(tet(tInd, 1), 0) += vol;
        //     m_vertexMasses(tet(tInd, 2), 0) += vol;
        //     m_vertexMasses(tet(tInd, 3), 0) += vol;
        // }
        // PD_PARALLEL_FOR
        // for (int vInd = 0; vInd < _N; vInd++) {
        //     m_vertexMasses(vInd, 0) = std::max(m_vertexMasses(vInd, 0), PDScalar(PD_MIN_MASS));
        // }
        // double totalMass = m_vertexMasses.sum();
        // spdlog::info("Average vertex mass: {}", totalMass * 1.0 / _N);
        // // Normalize vertex masses to integrate to 1 for numerical reasons
        // m_normalizationMass = 1.0 / totalMass;
        // spdlog::info("Normalize vertex mass: {}", m_normalizationMass);
        // m_vertexMasses *= m_normalizationMass * g_InteractState.hrpdParams.massPerUnitArea;
    }

    { // For uniform show data - Establish neighbourhood structure for tets or triangles
      // m_adjVerts1rd.resize(m_positions.rows());
      // m_adjVerts2rd.resize(m_positions.rows());
      // for (int i = 0; i < m_tets.rows(); i++) {
      //     for (int j = 0; j < 4; j++) {
      //         for (int k = j + 1; k < 4; k++) {
      //             m_adjVerts1rd[m_tets(i, j)].push_back(m_tets(i, k));
      //             m_adjVerts1rd[m_tets(i, k)].push_back(m_tets(i, j));
      //         }
      //     }
      // }
      // for (int v = 0; v < m_positions.rows(); v++) {
      //     for (auto nv1 : m_adjVerts1rd[v]) {
      //         {
      //             auto fd = std::find(m_adjVerts2rd[v].begin(), m_adjVerts2rd[v].end(), nv1);
      //             if (fd == m_adjVerts2rd[v].end()) {
      //                 m_adjVerts2rd[v].push_back(nv1);
      //             }
      //         }
      //         for (auto nv2 : m_adjVerts1rd[nv1]) {
      //             auto fd = std::find(m_adjVerts2rd[v].begin(), m_adjVerts2rd[v].end(), nv2);
      //             if (fd == m_adjVerts2rd[v].end()) {
      //                 m_adjVerts2rd[v].push_back(nv2);
      //             }
      //         }
      //     }
      // }
      // // Build tetsPerVertex
      // m_tetsPerVertex.clear();
      // m_tetsPerVertex.resize(m_positions.rows());
      // for (int i = 0; i < m_tets.rows(); i++) {
      //     for (int v = 0; v < 4; v++) {
      //         unsigned int vInd = m_tets(i, v);
      //         m_tetsPerVertex[vInd].push_back({ i, v });
      //     }
      // }
    }
}

void TetMesh::InitTetConstraints(PDScalar mu, PDScalar lambda)
{
    m_mu = mu;
    m_lambda = lambda;

    m_system_dim = m_positions.rows() * 3;

    { // Mass [3N x 3N]
        // -- Get vertex masses
        m_vertexMasses.setZero(m_positions.rows());
        auto& pos = m_positions;
        auto& tet = m_tets;
        PD_PARALLEL_FOR
        for (int tInd = 0; tInd < m_tets.rows(); tInd++) {
            Eigen::Matrix<PDScalar, 3, 3> edges;
            edges.col(0) = pos.row(tet(tInd, 1)) - pos.row(tet(tInd, 0));
            edges.col(1) = pos.row(tet(tInd, 2)) - pos.row(tet(tInd, 0));
            edges.col(2) = pos.row(tet(tInd, 3)) - pos.row(tet(tInd, 0));
            double vol = std::abs(edges.determinant()) / 6.0;
            vol /= 4.0;

            for (int k = 0; k < 4; k++) {
                PD_PARALLEL_ATOMIC
                m_vertexMasses(tet(tInd, k), 0) += vol;
            }
        }
        PD_PARALLEL_FOR
        for (int vInd = 0; vInd < m_positions.rows(); vInd++) {
            m_vertexMasses(vInd, 0) = std::max(m_vertexMasses(vInd, 0), PDScalar(PD_MIN_MASS));
        }
        double totalMass = m_vertexMasses.sum();
        // Normalize vertex masses to integrate to 1 for numerical reasons
        m_normalizationMass = 1.0 / totalMass;
        m_vertexMasses *= m_normalizationMass * g_InteractState.hrpdParams.massPerUnitArea;

        // -- Get mass matrix
        std::vector<PDSparseMatrixTriplet> massTris(m_system_dim);
        std::vector<PDSparseMatrixTriplet> massInvTris(m_system_dim);
        PD_PARALLEL_FOR
        for (int v = 0; v < m_system_dim; v++) {
            massTris[v] = { v, v, m_vertexMasses(v / 3) };
            massInvTris[v] = { v, v, 1.0 / m_vertexMasses(v / 3) };
        }
        m_mass_matrix.setFromTriplets(massTris.begin(), massTris.end());
        m_mass_matrix_inv.setFromTriplets(massInvTris.begin(), massInvTris.end());
        m_mass_matrix.makeCompressed();
        m_mass_matrix_inv.makeCompressed();
    }

    m_DmInvs.resize(m_tets.rows());
    m_Ws.resize(m_tets.rows());
    reduced_to_full(m_positions, m_positions_vec);
    reduced_to_full(m_velocities, m_velocities_vec);

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

        m_DmInvs[tInd] = edges.inverse();
        m_Ws[tInd] = 1.0 / 6.0 * std::abs(m_DmInvs[tInd].determinant());
    }
}

void TetMesh::EvalEnergy()
{
    m_E = 0;
    m_Es.resize(m_tets.rows());
    PD_PARALLEL_FOR
    for (int tInd = 0; tInd < m_tets.rows(); tInd++) {
        m_Es(tInd) = Get_element_energy(tInd);
    }
    m_E = m_Es.sum();
}

void TetMesh::EvalGradient()
{
    m_Grad.setZero(m_positions.rows() * 3);
    PD_PARALLEL_FOR
    for (int tInd = 0; tInd < m_tets.rows(); tInd++) {
        Eigen::Matrix<PDScalar, 12, 9> vec_dF_dx = Get_vec_dF_dx(tInd);
        Eigen::Matrix<PDScalar, 9, 1> vec_dPhi_dF = Get_vec_dPhi_dF(tInd);
        Eigen::Matrix<PDScalar, 12, 1> grad = vec_dF_dx.transpose() * vec_dPhi_dF;

        for (int i = 0; i < 12; i++) {
            PD_PARALLEL_ATOMIC
            m_Grad(m_tets(tInd, i / 3) + i % 3) += grad(i);
        }
    }
}

void TetMesh::EvalHessian(bool definitness_fix)
{
    std::vector<PDSparseMatrixTriplet> tris(m_positions.rows() * 3);
    int tris_idx = 0;

    PD_PARALLEL_FOR
    for (int tInd = 0; tInd < m_tets.rows(); tInd++) {
        Eigen::Matrix<PDScalar, 9, 12> vec_dF_dx = Get_vec_dF_dx(tInd);
        Eigen::Matrix<PDScalar, 9, 9> vec_d2Phi_dF2 = Get_vec_d2Phi_dF2(tInd);
        Eigen::Matrix<PDScalar, 12, 12> Hessian = vec_dF_dx.transpose() * vec_d2Phi_dF2 * vec_dF_dx;

        if (definitness_fix) { // Definitness fix
            Hessian = project_to_psd(Hessian);
        }

        // Assemble
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                // Assume x is flattened - [3N x 1]
                tris[tris_idx++] = PDSparseMatrixTriplet(m_tets(tInd, i / 3) * 3 + i % 3, m_tets(tInd, j / 3) * 3 + j % 3, Hessian(i, j));
            }
        }
    }

    m_H.setFromTriplets(tris.begin(), tris.end());
    m_H.makeCompressed();
}

// ===============================================================================

void TetMesh::reduced_to_full(Eigen::Matrix<PDScalar, -1, 3>& reduced, Eigen::Matrix<PDScalar, -1, 1>& full)
{
    full.resize(reduced.rows() * 3);
    PD_PARALLEL_FOR
    for (int i = 0; i < reduced.rows(); i++) {
        full(i * 3 + 0) = reduced.row(i).x();
        full(i * 3 + 1) = reduced.row(i).y();
        full(i * 3 + 2) = reduced.row(i).z();
    }
}

void TetMesh::full_to_reduced(Eigen::Matrix<PDScalar, -1, 3>& reduced, Eigen::Matrix<PDScalar, -1, 1>& full)
{
    PD_PARALLEL_FOR
    for (int i = 0; i < reduced.rows(); i++) {
        reduced.row(i).x() = full(i * 3 + 0);
        reduced.row(i).y() = full(i * 3 + 1);
        reduced.row(i).z() = full(i * 3 + 2);
    }
}

Eigen::Matrix<PDScalar, 3, 3> TetMesh::Get_deformation_gradient(int tInd)
{
    Eigen::Matrix<PDScalar, 3, 3> F;
    int v0 = m_tets(tInd, 0);
    int v1 = m_tets(tInd, 1);
    int v2 = m_tets(tInd, 2);
    int v3 = m_tets(tInd, 3);
    F.col(0) = m_positions.row(v1) - m_positions.row(v0);
    F.col(1) = m_positions.row(v2) - m_positions.row(v0);
    F.col(2) = m_positions.row(v3) - m_positions.row(v0);
    F *= m_DmInvs[tInd];
    return F;
}

PDScalar TetMesh::Get_element_energy(int tInd)
{
    EigenMatrix3 F = Get_deformation_gradient(tInd);

    PDScalar e_this = 0;
    switch (m_material_type) {
    case MATERIAL_TYPE_COROT: {
        // Eigen Jacobi SVD
        Eigen::JacobiSVD<EigenMatrix3> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        EigenMatrix3 U = svd.matrixU();
        EigenMatrix3 V = svd.matrixV();
        EigenVector3 SIGMA = svd.singularValues();

        { // signed_svd
            PDScalar detU = U.determinant();
            PDScalar detV = V.determinant();
            if (detU < 0) {
                U.block<3, 1>(0, 2) *= -1;
                SIGMA[2] *= -1;
            }
            if (detV < 0) {
                V.block<3, 1>(0, 2) *= -1;
                SIGMA[2] *= -1;
            }
        }

        EigenMatrix3 R = U * V.transpose();

        e_this = m_mu * (F - R).squaredNorm() + 0.5 * m_lambda * std::pow((R.transpose() * F).trace() - 3, 2);
    } break;
    case MATERIAL_TYPE_StVK: {
        EigenMatrix3 I = EigenMatrix3::Identity();
        EigenMatrix3 E = 0.5 * (F.transpose() * F - I);
        e_this = m_mu * E.squaredNorm() + 0.5 * m_lambda * std::pow(E.trace(), 2);
        // PDScalar J = F.determinant();
        // if (J < 1) {
        //     e_this += m_kappa / 12 * std::pow((1 - J) / 6, 3);
        // }
    } break;
    case MATERIAL_TYPE_NEOHOOKEAN: {
        EigenMatrix3 FtF = F.transpose() * F;
        PDScalar I1 = FtF.trace();
        PDScalar J = F.determinant();
        e_this = 0.5 * m_mu * (I1 - 3);
        PDScalar logJ;
        const PDScalar& J0 = m_neohookean_clamp_value;
        if (J > J0) {
            logJ = std::log(J);
            e_this += -m_mu * logJ + 0.5 * m_lambda * logJ * logJ;
        }
        else {
#ifdef LOGJ_QUADRATIC_EXTENSION
            PDScalar fJ = log(J0) + (J - J0) / J0 - 0.5 * std::pow((J / J0 - 1), 2);
#else
            PDScalar fJ = log(J0) + (J - J0) / J0;
#endif
            e_this += -m_mu * fJ + 0.5 * m_lambda * fJ * fJ;
        }
    } break;
    }

    e_this *= m_Ws[tInd];

    return e_this;
}

Eigen::Matrix<PDScalar, 9, 12> TetMesh::Get_vec_dF_dx(int tInd)
{
    Eigen::Matrix<PDScalar, 9, 12> vec_dF_dx = Eigen::Matrix<PDScalar, 9, 12>::Zero();
    auto& DmInv = m_DmInvs[tInd];

    PDScalar s0 = DmInv.col(0).sum();
    PDScalar s1 = DmInv.col(1).sum();
    PDScalar s2 = DmInv.col(2).sum();

    vec_dF_dx(0, 0) = vec_dF_dx(1, 1) = vec_dF_dx(2, 2) = -s0;
    vec_dF_dx(3, 0) = vec_dF_dx(4, 1) = vec_dF_dx(5, 2) = -s1;
    vec_dF_dx(6, 0) = vec_dF_dx(7, 1) = vec_dF_dx(8, 2) = -s2;

    vec_dF_dx(0, 3) = vec_dF_dx(1, 4) = vec_dF_dx(2, 5) = DmInv(0, 0);
    vec_dF_dx(3, 3) = vec_dF_dx(4, 4) = vec_dF_dx(5, 5) = DmInv(0, 1);
    vec_dF_dx(6, 3) = vec_dF_dx(7, 4) = vec_dF_dx(8, 5) = DmInv(0, 2);

    vec_dF_dx(0, 6) = vec_dF_dx(1, 7) = vec_dF_dx(2, 8) = DmInv(1, 0);
    vec_dF_dx(3, 6) = vec_dF_dx(4, 7) = vec_dF_dx(5, 8) = DmInv(1, 1);
    vec_dF_dx(6, 6) = vec_dF_dx(7, 7) = vec_dF_dx(8, 8) = DmInv(1, 2);

    vec_dF_dx(0, 9) = vec_dF_dx(1, 10) = vec_dF_dx(2, 11) = DmInv(2, 0);
    vec_dF_dx(3, 9) = vec_dF_dx(4, 10) = vec_dF_dx(5, 11) = DmInv(2, 1);
    vec_dF_dx(6, 9) = vec_dF_dx(7, 10) = vec_dF_dx(8, 11) = DmInv(2, 2);

    return vec_dF_dx;
}

Eigen::Matrix<PDScalar, 9, 1> TetMesh::Get_vec_dPhi_dF(int tInd)
{
    Eigen::Matrix<PDScalar, 9, 1> vec_dPhi_dF = Eigen::Matrix<PDScalar, 9, 1>::Zero();
    Eigen::Matrix<PDScalar, 3, 3> F = Get_deformation_gradient(tInd);

    if (m_material_type == MATERIAL_TYPE::MATERIAL_TYPE_NEOHOOKEAN) {
        PDScalar J = F.determinant();
        PDScalar logJ = std::log(J);
        Eigen::Matrix<PDScalar, 3, 1> f0 = F.col(0);
        Eigen::Matrix<PDScalar, 3, 1> f1 = F.col(1);
        Eigen::Matrix<PDScalar, 3, 1> f2 = F.col(2);

        auto _cross = [](Eigen::Matrix<PDScalar, 3, 1>& x, Eigen::Matrix<PDScalar, 3, 1>& y) {
            Eigen::Matrix<PDScalar, 3, 1> z;
            z.setZero();
            z(0) = x(1) * y(2) - x(2) * y(1);
            z(1) = x(2) * y(0) - x(0) * y(2);
            z(2) = x(0) * y(1) - x(1) * y(0);
            return z;
        };

        Eigen::Matrix<PDScalar, 3, 3> dJ_dF;
        dJ_dF.col(0) = _cross(f1, f2);
        dJ_dF.col(1) = _cross(f2, f0);
        dJ_dF.col(2) = _cross(f0, f1);

        Eigen::Matrix<PDScalar, 3, 3> dPhi_dF = m_mu * F - (m_mu / J) * dJ_dF + (m_lambda * logJ) / J * dJ_dF;
        vec_dPhi_dF = Eigen::Map<const Eigen::Matrix<PDScalar, 9, 1>>(dPhi_dF.data(), dPhi_dF.size());
    }

    return vec_dPhi_dF;
}

Eigen::Matrix<PDScalar, 9, 9> TetMesh::Get_vec_d2Phi_dF2(int tInd)
{
    Eigen::Matrix<PDScalar, 9, 9> vec_d2Phi_dF2 = Eigen::Matrix<PDScalar, 9, 9>::Zero();
    Eigen::Matrix<PDScalar, 3, 3> F = Get_deformation_gradient(tInd);

    if (m_material_type == MATERIAL_TYPE::MATERIAL_TYPE_NEOHOOKEAN) {
        PDScalar J = F.determinant();
        PDScalar logJ = std::log(J);
        Eigen::Matrix<PDScalar, 3, 1> f0 = F.col(0);
        Eigen::Matrix<PDScalar, 3, 1> f1 = F.col(1);
        Eigen::Matrix<PDScalar, 3, 1> f2 = F.col(2);

        auto _cross = [](Eigen::Matrix<PDScalar, 3, 1>& x, Eigen::Matrix<PDScalar, 3, 1>& y) {
            Eigen::Matrix<PDScalar, 3, 1> z;
            z.setZero();
            z(0) = x(1) * y(2) - x(2) * y(1);
            z(1) = x(2) * y(0) - x(0) * y(2);
            z(2) = x(0) * y(1) - x(1) * y(0);
            return z;
        };

        auto _hat = [](Eigen::Matrix<PDScalar, 3, 1>& v) {
            Eigen::Matrix<PDScalar, 3, 3> M;
            M.setZero();
            M(0, 1) = -v(2);
            M(0, 2) = v(1);
            M(1, 0) = v(2);
            M(1, 2) = -v(0);
            M(2, 0) = -v(1);
            M(2, 1) = v(0);
            return M;
        };

        Eigen::Matrix<PDScalar, 3, 3> dJ_dF;
        dJ_dF.col(0) = _cross(f1, f2);
        dJ_dF.col(1) = _cross(f2, f0);
        dJ_dF.col(2) = _cross(f0, f1);
        // g_j = vec(dJ_dF)
        Eigen::Matrix<PDScalar, 9, 1> g_j = Eigen::Map<const Eigen::Matrix<PDScalar, 9, 1>>(dJ_dF.data(), dJ_dF.size());

        Eigen::Matrix<PDScalar, 9, 9> d2J_dF2;
        d2J_dF2.setZero();
        d2J_dF2.block(0, 3, 3, 3) = -_hat(f2);
        d2J_dF2.block(0, 6, 3, 3) = _hat(f1);
        d2J_dF2.block(3, 0, 3, 3) = _hat(f2);
        d2J_dF2.block(3, 6, 3, 3) = -_hat(f0);
        d2J_dF2.block(6, 0, 3, 3) = -_hat(f1);
        d2J_dF2.block(6, 3, 3, 3) = _hat(f0);

        vec_d2Phi_dF2 += m_mu * Eigen::Matrix<PDScalar, 9, 9>::Identity();
        vec_d2Phi_dF2 += ((m_mu + m_lambda * (1 - logJ)) / (J * J)) * (g_j * g_j.transpose());
        vec_d2Phi_dF2 += ((m_lambda * logJ - m_mu) / J) * (d2J_dF2);
    }

    return vec_d2Phi_dF2;
}

// Matrix Projection onto Positive Semi-Definite Cone
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
TetMesh::project_to_psd(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& A)
{
    // https://math.stackexchange.com/q/2776803
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> eigensolver(A);
    if (eigensolver.info() != Eigen::Success) {
        // logger().error("unable to project matrix onto positive semi-definite cone"); // singleton for multithread
        spdlog::error("unable to project matrix onto positive semi-definite cone");
        throw std::runtime_error("unable to project matrix onto positive definite cone");
    }
    // Check if all eigen values are zero or positive.
    // The eigenvalues are sorted in increasing order.
    if (eigensolver.eigenvalues()[0] >= 0.0) {
        return A;
    }
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(eigensolver.eigenvalues());
    // Save a little time and only project the negative values
    for (int i = 0; i < A.rows(); i++) {
        if (D.diagonal()[i] < 0.0) {
            D.diagonal()[i] = 0.0;
        }
        else {
            break;
        }
    }
    return eigensolver.eigenvectors() * D * eigensolver.eigenvectors().transpose();
}

}

}