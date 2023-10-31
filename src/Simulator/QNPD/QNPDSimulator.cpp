#include "QNPDSimulator.hpp"
#include "QNPDConstraint.hpp"
#include "Simulator/PDTypeDef.hpp"
#include "UI/InteractState.hpp"
#include "Util/Profiler.hpp"
#include "Util/StoreData.hpp"
#include "Util/Timer.hpp"

#include "spdlog/spdlog.h"
#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

extern UI::InteractState g_InteractState;
extern Util::Profiler g_StepProfiler;

namespace PD {

QNPDSimulator::QNPDSimulator(std::shared_ptr<QNPDTetMesh> tetMesh)
    : m_mesh(tetMesh)
{
    m_processing_collision = true;

    m_y.resize(m_mesh->m_system_dimension);
    m_external_force.resize(m_mesh->m_system_dimension);

    LoadParamsAndApply();

    spdlog::info(">>>After QNPDSimulator::QNPDSimulator()");
}

QNPDSimulator::~QNPDSimulator()
{
    clearConstraints();
}

void QNPDSimulator::LoadParamsAndApply()
{
    m_dt = g_InteractState.timeStep;
    m_dt2 = m_dt * m_dt;
    m_dt2inv = 1.0 / m_dt2;
    m_mesh->m_total_mass = g_InteractState.hrpdParams.massPerUnitArea;

    auto qnpdParams = g_InteractState.qnpdParams;

    if (qnpdParams.materialType == "MATERIAL_TYPE_StVK") {
        m_material_type = MATERIAL_TYPE_StVK;
    }
    else if (qnpdParams.materialType == "MATERIAL_TYPE_COROT") {
        m_material_type = MATERIAL_TYPE_COROT;
    }
    else if (qnpdParams.materialType == "MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG") {
        m_material_type = MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG;
    }
    else {
        spdlog::warn("MATERIAL [{}] NOT FOUND - USE DEFAULT: MATERIAL_TYPE_StVK", g_InteractState.qnpdParams.materialType);
        m_material_type = MATERIAL_TYPE_StVK;
    }
    m_stiffness_attachment = qnpdParams.stiffness_attachment * 100;
    m_stiffness_stretch = qnpdParams.stiffness_stretch * 100;
    m_stiffness_bending = qnpdParams.stiffness_bending * 100;
    m_stiffness_kappa = qnpdParams.stiffness_kappa * 100;
    m_stiffness_laplacian = qnpdParams.stiffness_laplacian * 100;
    m_damping_coefficient = qnpdParams.damping_coefficient;

    m_lbfgs.m = qnpdParams.lbfgs_m;

    m_ls.enable_line_search = qnpdParams.ls_enable_line_search;
    m_ls.enable_exact_search = qnpdParams.ls_enable_exact_search;
    m_ls.step_size = qnpdParams.ls_step_size;
    m_ls.alpha = qnpdParams.ls_alpha;
    m_ls.beta = qnpdParams.ls_beta;

    setupConstraints();
}

void QNPDSimulator::Reset()
{
    m_frameCount = 0;

    m_mesh->m_total_mass = 1;
    m_mesh->ResetPositions();

    setupConstraints();

    // lbfgs
    m_lbfgs.restart_every_frame = false;
    m_lbfgs.need_update_H0 = true;

    {
        m_OpManager.Reset();
    }
}

void QNPDSimulator::Step()
{
    PROFILE_STEP("QNPD Step");
    calculateExternalForce(); // maybe no need calc every step, but it doesnt cost much

    PDScalar old_h = m_dt;
    m_dt /= g_InteractState.substeps;

    PDVector x;
    for (int substep_i = 0; substep_i < g_InteractState.substeps; substep_i++) { // 1 iteration in fact
        PROFILE_STEP("1 Substep");
        {
            PROFILE_STEP("INERTIA");
            // update inertia term - INTEGRATION_IMPLICIT_EULER
            m_y = m_mesh->m_current_positions_vec + m_dt * m_mesh->m_current_velocities_vec + m_dt2 * m_mesh->m_inv_mass_matrix * m_external_force;
            x = m_y;
        }

        // update iteration
        if (m_lbfgs.restart_every_frame == true) {
            m_lbfgs.need_update_H0 = true;
        }

        bool converge = false;
        m_lbfgs.is_first_iteration = true;
        // TICK(LBFGS_TOTAL_ITERATION);
        for (m_current_iteration = 0; !converge && m_current_iteration < g_InteractState.numIterations; m_current_iteration++) {
            PROFILE_STEP("ITERATION");
            // Collision handle
            if (m_processing_collision) {
                // Collision Detection every iteration
                collisionDetection(x);
            }

            // Iteration
            switch (m_optimization_method) {
            case OPTIMIZATION_METHOD_GRADIENT_DESCENT:
                converge = performGradientDescentOneIteration(x);
                break;
            case OPTIMIZATION_METHOD_NEWTON:
                converge = performNewtonsMethodOneIteration(x);
                break;
            case OPTIMIZATION_METHOD_LBFGS:
                converge = performLBFGSOneIteration(x);
                break;
            default:
                break;
            }
            // spdlog::info(">>> QNPDSimulator::step - m_current_iteration = {}, converge = {}", m_current_iteration, converge);

            // Others
            m_lbfgs.is_first_iteration = false;
        }
        // TOCK(LBFGS_TOTAL_ITERATION);

        {
            PROFILE_STEP("UPDATE POS");
            // update positions and velocities
            m_mesh->m_previous_velocities_vec = m_mesh->m_current_velocities_vec;
            m_mesh->m_previous_positions_vec = m_mesh->m_current_positions_vec;
            m_mesh->m_current_velocities_vec = (x - m_mesh->m_current_positions_vec) / m_dt;
            m_mesh->m_current_positions_vec = x;
        }

        // damping
        if (std::abs(m_damping_coefficient) >= EPSILON) {
            m_mesh->m_current_velocities_vec *= 1 - m_damping_coefficient;
        }
    }

    // m_cur_positions = Eigen::Map<PDPositions>(m_mesh->m_current_positions_vec.data(), m_mesh->m_vertices_number, 3);
    PD_PARALLEL_FOR
    for (int i = 0; i < m_mesh->m_vertices_number; i++) {
        for (int j = 0; j < 3; j++) {
            m_mesh->m_positions(i, j) = m_mesh->m_current_positions_vec[3 * i + j];
        }
    }
    // spdlog::info(">>> m_cur_positions size = ({}, {})", m_cur_positions.rows(), m_cur_positions.cols());

    {
        // static int timestep = 0;
        // std::ofstream f;
        // f.open("./debug/QNPD/m_y/" + std::to_string(timestep));
        // f << m_y;
        // f.close();

        // Util::storeData(m_mesh->m_positions, m_mesh->m_triangles, "./debug/QNPD/objs/" + std::to_string(timestep) + ".obj", true);
        // timestep++;
    }

    { // Operation objects step
        PROFILE_STEP("OP_OBJECTS_STEP");
        // if (_debug) spdlog::info("> HRPDSimulator::Step() - OP_OBJECTS_STEP");
        m_OpManager.Step(m_dt);
    }

    ms_x = x;

    m_dt = old_h;

    m_frameCount++;
    // spdlog::info(">>>After QNPDSimulator::step()");
}

void QNPDSimulator::UpdateTimeStep()
{
    m_dt = g_InteractState.timeStep;
    m_dt2 = m_dt * m_dt;
    m_dt2inv = 1.0 / m_dt2;
}

// -------------------------------------------------------------------------------------------------------------------

void QNPDSimulator::setMaterialProperty(std::vector<Constraint*>& constraints)
{
    PD_PARALLEL_FOR
    for (int i = 0; i < constraints.size(); i++) {
        constraints[i]->SetMaterialProperty(m_material_type, m_stiffness_stretch, m_stiffness_bending, m_stiffness_kappa, m_stiffness_laplacian);
    }
    m_precomputing_flag = false;
    m_prefactorization_flag = false;
    m_prefactorization_flag_newton = false;
}

void QNPDSimulator::clearConstraints()
{
    for (unsigned int i = 0; i < m_constraints.size(); ++i) {
        delete m_constraints[i];
    }
    m_constraints.clear();
}

void QNPDSimulator::setupConstraints()
{
    clearConstraints();

    // For MESH_TYPE_TET
    PDScalar total_volume = 0;
    std::vector<PDSparseMatrixTriplet> mass_triplets, mass_1d_triplets;
    PDVector& x = m_mesh->m_current_positions_vec;
    PDVector mass_vec;
    mass_vec.setZero(m_mesh->m_vertices_number, 1);
    for (size_t i = 0; i < m_mesh->m_tets.rows(); i++) {
        auto tet = m_mesh->m_tets.row(i);
        QNPDTetConstraint* c = new QNPDTetConstraint(tet[0], tet[1], tet[2], tet[3], x);
        m_constraints.push_back(c);
        PDScalar cur_vol;
        cur_vol = c->SetMassMatrix(mass_triplets, mass_1d_triplets);
        // total_volume += c->SetMassMatrix(mass_triplets, mass_1d_triplets);
        total_volume += cur_vol;
        for (int k = 0; k < 4; k++) {
            mass_vec(c->m_p[k]) += 0.25 * cur_vol;
        }
    }
    // spdlog::info(">>> m_mesh->m_tets size = {}", m_mesh->m_tets.rows());
    // std::cout << m_mesh->m_tets.row(0) << std::endl;

    { // Set mass
        m_mesh->m_mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
        m_mesh->m_mass_matrix_1d.setFromTriplets(mass_1d_triplets.begin(), mass_1d_triplets.end());
        m_mesh->m_mass_matrix = m_mesh->m_mass_matrix * (m_mesh->m_total_mass / total_volume);
        m_mesh->m_mass_matrix_1d = m_mesh->m_mass_matrix_1d * (m_mesh->m_total_mass / total_volume);

        // for (int i = 0; i < mass_vec.rows(); i++) mass_vec(i) *= (m_mesh->m_total_mass / total_volume);
        mass_vec = mass_vec * (m_mesh->m_total_mass / total_volume);
        spdlog::info("Total volume: {}, Normalize vertex mass: {}", total_volume, (m_mesh->m_total_mass / total_volume));
        // spdlog::info(">>>After QNPDSimulator::setupConstraints() - ok 2");

        std::vector<PDSparseMatrixTriplet> mass_inv_triplets, mass_inv_1d_triplets;
        for (size_t i = 0; i < m_mesh->m_mass_matrix.rows(); i++) {
            PDScalar mi = m_mesh->m_mass_matrix.coeff(i, i);
            PDScalar mi_inv;
            if (std::abs(mi) > 1e-12) {
                mi_inv = 1.0 / mi;
            }
            else {
                // ugly ugly!
                m_mesh->m_mass_matrix.coeffRef(i, i) = 1e-12;
                mi_inv = 1e12;
            }
            mass_inv_triplets.push_back(PDSparseMatrixTriplet(i, i, mi_inv));
        }
        for (size_t i = 0; i != m_mesh->m_mass_matrix_1d.rows(); i++) {
            PDScalar mi = m_mesh->m_mass_matrix_1d.coeff(i, i);
            PDScalar mi_inv;
            if (std::abs(mi) > 1e-12) {
                mi_inv = 1.0 / mi;
            }
            else {
                // ugly ugly!
                m_mesh->m_mass_matrix_1d.coeffRef(i, i) = 1e-12;
                mi_inv = 1e12;
                mass_vec[i] = 1e-12;
            }
            mass_inv_1d_triplets.push_back(PDSparseMatrixTriplet(i, i, mi_inv));
        }
        m_mesh->m_inv_mass_matrix.setFromTriplets(mass_inv_triplets.begin(), mass_inv_triplets.end());
        m_mesh->m_inv_mass_matrix_1d.setFromTriplets(mass_inv_1d_triplets.begin(), mass_inv_1d_triplets.end());
    }
    setMaterialProperty(m_constraints);

    { // debug
      // std::ofstream f;
      // f.open("./debug/QNPD/mesh_info");
      // f << fmt::format("V nums = {}, Tet nums = {}\n", m_mesh->m_vertices_number, m_mesh->m_tets.rows());
      // for (int i = 0; i < m_mesh->m_positions.rows(); i++) {
      //     f << fmt::format("V {}: ", i);
      //     f << m_mesh->m_positions.row(i);
      //     f << "\n";
      // }
      // for (int i = 0; i < m_mesh->m_tets.rows(); i++) {
      //     f << fmt::format("Tet {}: ", i);
      //     f << m_mesh->m_tets.row(i);
      //     f << "\n";
      // }
      // f.close();

        // f.open("./debug/QNPD/mass_vec");
        // f << fmt::format("Size = ({}, {})\n", mass_vec.rows(), mass_vec.cols());
        // f << mass_vec;
        // f.close();

        // PDMatrix mass_1d = m_mesh->m_mass_matrix_1d;
        // f.open("./debug/QNPD/mass_1d");
        // f << fmt::format("Size = ({}, {})\n", mass_1d.rows(), mass_1d.cols());
        // f << mass_1d;
        // f.close();

        // PDMatrix mass_3d = m_mesh->m_mass_matrix;
        // f.open("./debug/QNPD/mass_3d");
        // f << fmt::format("Size = ({}, {})\n", mass_3d.rows(), mass_3d.cols());
        // f << mass_3d;
        // f.close();
    }

    spdlog::info(">>>After QNPDSimulator::setupConstraints()");
}

void QNPDSimulator::calculateExternalForce()
{
    PROFILE_STEP("Calc F_ext");
    m_external_force.resize(m_mesh->m_system_dimension);
    m_external_force.setZero();

    // gravity
    PD_PARALLEL_FOR
    for (int i = 0; i < m_mesh->m_vertices_number; ++i) {
        m_external_force[3 * i + 1] += -g_InteractState.qnpdParams.gravityConstant;
    }

    m_external_force = m_mesh->m_mass_matrix * m_external_force;
}

// --------------------------------------------------------------------

void QNPDSimulator::collisionDetection(const PDVector& x)
{
    PROFILE_STEP("CollisionDetect");
    m_collision_constraints.resize(m_mesh->m_vertices_number, { -1, -1, { 0, 0, 0 }, { 0, 0, 0 } });
    PD_PARALLEL_FOR
    for (int i = 0; i < m_mesh->m_vertices_number; i++) {
        m_collision_constraints[i].m_p0 = -1;
    }

    // User interaction - Drag
    if (g_InteractState.draggingState.isDragging) {
        EigenVector3 surface_point;
        EigenVector3 normal{ 0, 0, 0 };
        int v = g_InteractState.draggingState.vertex;
        auto add = (1.0 / m_dt2) * g_InteractState.draggingState.force.cast<PDScalar>();
        surface_point = { x[3 * v] + add[0], x[3 * v + 1] + add[1], x[3 * v + 2] + add[2] };
        normal = add;
        // m_collision_constraints[v] = QNPDPenaltyConstraint(1e-4, v, surface_point, normal);
        m_collision_constraints[v] = QNPDPenaltyConstraint(1, v, surface_point, normal);
        // spdlog::info(">>> collisionDetection - Add Drag - v = {}", v);
    }

    // Floor Constraint
    // {
    //     PD_PARALLEL_FOR
    //     for (int i = 0; i < m_mesh->m_vertices_number; i++) {
    //         if (x[3 * i + 1] < 0) {
    //             EigenVector3 surface_point = { x[3 * i], 0, x[3 * i + 2] };
    //             EigenVector3 normal{ 0, 1, 0 };
    //             m_collision_constraints[i] = (QNPDPenaltyConstraint(1e2, i, surface_point, normal));
    //         }
    //     }
    // }

    // Collision handle
    {
        PD_PARALLEL_FOR
        for (int i = 0; i < m_mesh->m_vertices_number; i++) {
            // -- Floor
            if (g_InteractState.isFloorActive) {
                if (x[3 * i + 1] < 0) {
                    EigenVector3 surface_point = { x[3 * i], 0, x[3 * i + 2] };
                    EigenVector3 normal{ 0, 1, 0 };
                    m_collision_constraints[i] = (QNPDPenaltyConstraint(1e2, i, surface_point, normal));
                }
            }
            // -- Collision
            for (auto& obj : m_OpManager.m_operationObjects) {
                auto colObj = std::dynamic_pointer_cast<CollisionObject>(obj);
                if (colObj != nullptr) {
                    PD3dVector posV = { x[3 * i], x[3 * i + 1], x[3 * i + 2] };
                    EigenVector3 normal = { 0, 0, 0 };
                    if (colObj->ResolveCollision(posV, normal)) {
                        // spdlog::info(">>> Collision detect:");
                        // std::cout << posV.transpose() << std::endl;

                        // pos.row(v) = posV.transpose();
                        EigenVector3 surface_point = posV;
                        PD3dVector curV = { x[3 * i], x[3 * i + 1], x[3 * i + 2] };
                        // EigenVector3 normal = (posV - curV).normalized();
                        m_collision_constraints[i] = (QNPDPenaltyConstraint(1e2, i, surface_point, normal));
                    }
                }
            }
            // -- Grip
            // for (auto& obj : m_OpManager.m_operationObjects) {
            //     auto gripObj = std::dynamic_pointer_cast<GripObject>(obj);
            //     if (gripObj != nullptr) {
            //         PD3dVector posV = pos.row(v);
            //         PD3dVector prevPosV = m_mesh->m_positions.row(m_subspaceBuilder.m_usedVertices[v]);
            //         if (gripObj->ResolveGrip(posV)) {
            //             gripCorrection = true;
            //             pos.row(v) = prevPosV.transpose();
            //         }
            //     }
            // }
        }
    }
}

// ================================= Iteration =================================

bool QNPDSimulator::performGradientDescentOneIteration(PDVector& x)
{
    // evaluate gradient direction
    PDVector gradient;
    evaluateGradient(x, gradient);

#ifdef ENABLE_MATLAB_DEBUGGING
    g_debugger->SendVector(gradient, "g");
#endif

    if (gradient.norm() < EPSILON)
        return true;

    // assign descent direction
    // PDVector descent_dir = -m_mesh->m_inv_mass_matrix*gradient;
    PDVector descent_dir = -gradient;

    // line search
    PDScalar step_size = lineSearch(x, gradient, descent_dir);

    // update x
    x = x + descent_dir * step_size;

    // report convergence
    if (step_size < EPSILON)
        return true;
    else
        return false;
}

bool QNPDSimulator::performNewtonsMethodOneIteration(PDVector& x)
{
    PROFILE_STEP("NewtonOneIteration");
    // spdlog::info(">>> QNPDSimulator::performGradientDescentOneIteration()");
    // TimerWrapper timer; timer.Tic();
    // evaluate gradient direction
    PDVector gradient;
    {
        PROFILE_STEP("EvalGrad");
        evaluateGradient(x, gradient, true);
    }
    // spdlog::info(">>> QNPDSimulator::performGradientDescentOneIteration() - After evaluateGradient()");
    // QSEvaluateGradient(x, gradient, m_ss->m_quasi_static);
#ifdef ENABLE_MATLAB_DEBUGGING
    g_debugger->SendVector(gradient, "g");
#endif

    // timer.TocAndReport("evaluate gradient", m_verbose_show_converge);
    // timer.Tic();

    // evaluate hessian matrix
    PDSparseMatrix hessian_1;
    {
        PROFILE_STEP("EvalHessian");
        evaluateHessian(x, hessian_1);
    }
    // spdlog::info(">>> QNPDSimulator::performGradientDescentOneIteration() - After evaluateHessian()");
    // PDSparseMatrix hessian_2;
    // evaluateHessianSmart(x, hessian_2);

    PDSparseMatrix& hessian = hessian_1;

#ifdef ENABLE_MATLAB_DEBUGGING
    g_debugger->SendSparseMatrix(hessian_1, "H");
    // g_debugger->SendSparseMatrix(hessian_2, "H2");
#endif

    // timer.TocAndReport("evaluate hessian", m_verbose_show_converge);
    // timer.Tic();
    PDVector descent_dir;

    {
        PROFILE_STEP("LinearSolve");
        linearSolve(descent_dir, hessian, gradient);
    }
    // spdlog::info(">>> QNPDSimulator::performGradientDescentOneIteration() - After linearSolve()");
    descent_dir = -descent_dir;

    // timer.TocAndReport("solve time", m_verbose_show_converge);
    // timer.Tic();

    // line search
    PDScalar step_size = 1;
    {
        PROFILE_STEP("LineSearch");
        step_size = lineSearch(x, gradient, descent_dir);
    }
    // spdlog::info(">>> QNPDSimulator::performGradientDescentOneIteration() - After lineSearch()");
    // if (step_size < EPSILON)
    //{
    //	std::cout << "correct step size to 1" << std::endl;
    //	step_size = 1;
    // }
    //  update x
    x = x + descent_dir * step_size;

    // if (step_size < EPSILON)
    //{
    //	printVolumeTesting(x);
    // }

    // timer.TocAndReport("line search", m_verbose_show_converge);
    // timer.Toc();
    // std::cout << "newton: " << timer.Duration() << std::endl;

    if (-descent_dir.dot(gradient) < EPSILON_SQUARE)
        return true;
    else
        return false;
}

bool QNPDSimulator::performLBFGSOneIteration(PDVector& x) // [###]
{
    PROFILE_STEP("LBFGSOneIteration");
    bool converged = false;
    PDScalar current_energy;
    PDVector gf_k;

    if (m_lbfgs.is_first_iteration || !m_ls.enable_line_search) {
        current_energy = evaluateEnergyAndGradient(x, gf_k);
    }
    else {
        current_energy = m_ls.prefetched_energy;
        gf_k = m_ls.prefetched_gradient;
    }

    if (m_lbfgs.is_first_iteration) { // first iteration
        // clear sk and yk and alpha_k
        m_lbfgs.y_queue.clear();
        m_lbfgs.s_queue.clear();

        // prefactorize H0_LAPLACIAN
        if (m_lbfgs.need_update_H0) {
            prefactorize();
            m_lbfgs.need_update_H0 = false;
        }

        // store x, gf_k before wipeout
        m_lbfgs.last_x = x;
        m_lbfgs.last_gradient = gf_k;

        // first iteration
        PDVector r;
        LBFGSKernelLinearSolve(r, gf_k, 1);

        PDVector p_k = -r;
        if (-p_k.dot(gf_k) < EPSILON_SQUARE || p_k.norm() / x.norm() < LARGER_EPSILON) {
            converged = true;
        }

        // line search
        PDScalar alpha_k = linesearchWithPrefetchedEnergyAndGradientComputing(
            x, current_energy, gf_k, p_k, m_ls.prefetched_energy, m_ls.prefetched_gradient);

        // update
        x += alpha_k * p_k;

        // spdlog::info(">>> QNPDSimulator::performLBFGSOneIteration - After first iteration");
    }
    else { // not first iteration
        // enqueue stuff
        PDVector s_k = x - m_lbfgs.last_x;
        PDVector y_k = gf_k - m_lbfgs.last_gradient;
        if (m_lbfgs.s_queue.size() > m_lbfgs.m) {
            m_lbfgs.s_queue.pop_back(), m_lbfgs.y_queue.pop_back();
        }
        m_lbfgs.s_queue.push_front(s_k);
        m_lbfgs.y_queue.push_front(y_k);

        // store x, gf_k before wipeout
        m_lbfgs.last_x = x;
        m_lbfgs.last_gradient = gf_k;
        int queue_visit_upper_bound = std::min(m_lbfgs.m, (int)m_lbfgs.s_queue.size());

        PDVector q = gf_k;

        std::vector<PDScalar> rho;
        std::vector<PDScalar> alpha;
        { // loop 1 of l-BFGS
            PDScalar s_i, y_i;
            for (int i = 0; i < queue_visit_upper_bound; i++) {
                PDScalar yi_dot_si = m_lbfgs.y_queue[i].dot(m_lbfgs.s_queue[i]);
                if (yi_dot_si < EPSILON_SQUARE) return true;
                PDScalar rho_i = 1.0 / yi_dot_si;
                rho.push_back(rho_i);
                alpha.push_back(rho[i] * m_lbfgs.s_queue[i].dot(q));
                q = q - alpha[i] * m_lbfgs.y_queue[i];
            }
        }
        // compute H0 * q
        PDVector r;
        // compute the scaling parameter on the fly
        PDScalar scaling_parameter = (s_k.transpose() * y_k).trace() / (y_k.transpose() * y_k).trace();
        if (scaling_parameter < EPSILON) // should not be negative
        {
            scaling_parameter = EPSILON;
        }
        LBFGSKernelLinearSolve(r, q, scaling_parameter);
        { // loop 2 of l-BFGS
            for (int i = queue_visit_upper_bound - 1; i >= 0; i--) {
                PDScalar beta = rho[i] * m_lbfgs.y_queue[i].dot(r);
                r = r + m_lbfgs.s_queue[i] * (alpha[i] - beta);
            }
        }
        // update
        PDVector p_k = -r;
        if (-p_k.dot(gf_k) < EPSILON_SQUARE || p_k.norm() / x.norm() < LARGER_EPSILON) {
            converged = true;
        }

        // line search
        PDScalar alpha_k = linesearchWithPrefetchedEnergyAndGradientComputing(
            x, current_energy, gf_k, p_k, m_ls.prefetched_energy, m_ls.prefetched_gradient);

        x += alpha_k * p_k;

        // spdlog::info(">>> QNPDSimulator::performLBFGSOneIteration - After iteration");
    }

    return converged;
}

// ================================= Iteration =================================

void QNPDSimulator::evaluateGradient(const PDVector& x, PDVector& gradient, bool enable_omp)
{
    PDVector gradient_pure(gradient.rows(), gradient.cols());
    PDScalar h_square = m_dt * m_dt;
    switch (m_integration_method) {
    case INTEGRATION_QUASI_STATICS:
        evaluateGradientPureConstraint(x, m_external_force, gradient_pure);
        gradient = gradient_pure - m_external_force;
        break; // DO NOTHING
    case INTEGRATION_IMPLICIT_EULER:
        evaluateGradientPureConstraint(x, m_external_force, gradient_pure);
        gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square * gradient_pure;
        break;
    case INTEGRATION_IMPLICIT_BDF2:
        evaluateGradientPureConstraint(x, m_external_force, gradient_pure);
        gradient = m_mesh->m_mass_matrix * (x - m_y) + (h_square * 4.0 / 9.0) * gradient_pure;
        break;
    case INTEGRATION_IMPLICIT_MIDPOINT:
        evaluateGradientPureConstraint((x + m_mesh->m_current_positions_vec) / 2, m_external_force, gradient_pure);
        gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 2 * (gradient_pure);
        break;
    case INTEGRATION_IMPLICIT_NEWMARK_BETA:
        evaluateGradientPureConstraint(x, m_external_force, gradient_pure);
        gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 4 * (gradient_pure + m_z);
        break;
    }

    static int ct = 0;
    if (ct < 100) { // debug
        std::ofstream f;
        f.open("./debug/QNPD/grad_pure/grad_pure_iter_" + std::to_string(ct));
        f << fmt::format("Size = ({}, {})\n", gradient_pure.rows(), gradient_pure.cols());
        f << gradient_pure;
        f.close();

        f.open("./debug/QNPD/grad/grad_iter_" + std::to_string(ct));
        f << fmt::format("Size = ({}, {})\n", gradient.rows(), gradient.cols());
        f << gradient;
        f.close();

        ct++;
    }
}

void QNPDSimulator::evaluateGradientPureConstraint(const PDVector& x, const PDVector& f_ext, PDVector& gradient)
{
    gradient.resize(m_mesh->m_system_dimension);
    gradient.setZero();

    if (!m_enable_openmp) {
        // constraints single thread
        for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it) {
            (*it)->EvaluateGradient(x, gradient);
        }
    }
    else {
        // constraints omp
        int i;
#pragma omp parallel
        {
#pragma omp for
            for (i = 0; i < m_constraints.size(); i++) {
                m_constraints[i]->EvaluateGradient(x);
            }
        }

        for (i = 0; i < m_constraints.size(); i++) {
            m_constraints[i]->GetGradient(gradient);
        }
    }

    // hardcoded collision plane
    if (m_processing_collision) {
        PDVector gc;

        evaluateGradientCollision(x, gc);

        gradient += gc;
    }
}

void QNPDSimulator::evaluateGradientCollision(const PDVector& x, PDVector& gradient)
{
    gradient.resize(m_mesh->m_system_dimension);
    gradient.setZero();

    if (!m_enable_openmp) {
        // constraints single thread
        for (std::vector<QNPDPenaltyConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it) {
            it->EvaluateGradient(x, gradient);
        }
    }
    else {
        // constraints omp
        int i;
#pragma omp parallel
        {
#pragma omp for
            for (i = 0; i < m_collision_constraints.size(); i++) {
                m_collision_constraints[i].EvaluateGradient(x);
            }
        }

        for (i = 0; i < m_collision_constraints.size(); i++) {
            m_collision_constraints[i].GetGradient(gradient);
        }
    }
}

PDScalar QNPDSimulator::evaluateEnergyAndGradient(const PDVector& x, PDVector& gradient)
{
    PROFILE_STEP("EVAL_E&grad");
    PDScalar h_square = m_dt2;
    PDScalar energy_pure_constraints, energy;
    PDScalar inertia_term = 0.5 * (x - m_y).transpose() * m_mesh->m_mass_matrix * (x - m_y);

    // INTEGRATION_IMPLICIT_EULER
    energy_pure_constraints = evaluateEnergyAndGradientPureConstraint(x, gradient);
    energy = inertia_term + h_square * energy_pure_constraints;
    gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square * gradient;

    return energy;
}

PDScalar QNPDSimulator::evaluateEnergyAndGradientPureConstraint(const PDVector& x, PDVector& gradient)
{
    PDScalar energy = 0.0;
    gradient.resize(m_mesh->m_system_dimension);
    gradient.setZero();

    {
        int i;
        PD_PARALLEL_FOR
        for (i = 0; i < m_constraints.size(); i++) {
            m_constraints[i]->EvaluateEnergyAndGradient(x);
        }

        // collect the results in a sequential way
        for (i = 0; i < m_constraints.size(); i++) {
            energy += m_constraints[i]->GetEnergyAndGradient(gradient);
        }
    }

    // hardcoded collision plane
    if (m_processing_collision) {
        PDVector gc;
        energy += evaluateEnergyAndGradientCollision(x, gc);
        gradient += gc;
    }

    return energy;
}

PDScalar QNPDSimulator::evaluateEnergyAndGradientCollision(const PDVector& x, PDVector& gradient)
{
    PDScalar energy = 0.0;
    gradient.resize(m_mesh->m_system_dimension);
    gradient.setZero();

    {
        // constraints omp
        int i;
        PD_PARALLEL_FOR
        for (i = 0; i < m_collision_constraints.size(); i++) {
            m_collision_constraints[i].EvaluateEnergyAndGradient(x);
        }

        // collect the results in a sequential way
        for (i = 0; i < m_collision_constraints.size(); i++) {
            energy += m_collision_constraints[i].GetEnergyAndGradient(gradient);
        }
    }

    // spdlog::info(">>> QNPDSimulator::evaluateEnergyAndGradientCollision() - energy = {}", energy);

    return energy;
}

void QNPDSimulator::evaluateLaplacianPureConstraint(PDSparseMatrix& laplacian_matrix)
{
    laplacian_matrix.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
    std::vector<PDSparseMatrixTriplet> l_triplets(16 * 9 * m_constraints.size());

    PD_PARALLEL_FOR
    for (int i = 0; i < m_constraints.size(); i++) {
        m_constraints[i]->EvaluateWeightedLaplacian(l_triplets, i);
    }

    laplacian_matrix.setFromTriplets(l_triplets.begin(), l_triplets.end());
    laplacian_matrix.makeCompressed();
}

void QNPDSimulator::evaluateLaplacianPureConstraint1D(PDSparseMatrix& laplacian_matrix_1d)
{
    laplacian_matrix_1d.resize(m_mesh->m_vertices_number, m_mesh->m_vertices_number);
    std::vector<PDSparseMatrixTriplet> l_1d_triplets(16 * m_constraints.size());

    PD_PARALLEL_FOR
    for (int i = 0; i < m_constraints.size(); i++) {
        m_constraints[i]->EvaluateWeightedLaplacian1D(l_1d_triplets, i);
    }
    // for (int i = 0; i < m_constraints.size(); i++) {
    //     m_constraints[i]->EvaluateWeightedLaplacian1D(l_1d_triplets);
    // }

    laplacian_matrix_1d.setFromTriplets(l_1d_triplets.begin(), l_1d_triplets.end());
    laplacian_matrix_1d.makeCompressed();
}

void QNPDSimulator::evaluateHessian(const PDVector& x, PDSparseMatrix& hessian_matrix)
{
    PDSparseMatrix hessian_pure(hessian_matrix.rows(), hessian_matrix.cols());
    PDScalar h_square = m_dt * m_dt;
    switch (m_integration_method) {
    case INTEGRATION_QUASI_STATICS:
        evaluateHessianPureConstraint(x, hessian_pure);
        hessian_matrix = hessian_pure;
        break; // DO NOTHING
    case INTEGRATION_IMPLICIT_EULER:
        evaluateHessianPureConstraint(x, hessian_pure);
        hessian_matrix = m_mesh->m_mass_matrix + h_square * hessian_pure;
        break;
    case INTEGRATION_IMPLICIT_BDF2:
        evaluateHessianPureConstraint(x, hessian_pure);
        hessian_matrix = m_mesh->m_mass_matrix + h_square * 4.0 / 9.0 * hessian_pure;
        break;
    case INTEGRATION_IMPLICIT_MIDPOINT:
        evaluateHessianPureConstraint((x + m_mesh->m_current_positions_vec) / 2, hessian_pure);
        hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_pure;
        break;
    case INTEGRATION_IMPLICIT_NEWMARK_BETA:
        evaluateHessianPureConstraint(x, hessian_pure);
        hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_pure;
        break;
    }

    static int ct = 0;
    if (ct < 100) { // debug

        std::ofstream f;
        f.open("./debug/QNPD/hessian_pure/hessian_pure_iter_" + std::to_string(ct));
        f << fmt::format("Size = ({}, {})\n", hessian_pure.rows(), hessian_pure.cols());
        f << hessian_pure;
        f.close();

        f.open("./debug/QNPD/hessian/hessian_iter_" + std::to_string(ct));
        f << fmt::format("Size = ({}, {})\n", hessian_matrix.rows(), hessian_matrix.cols());
        f << hessian_matrix;
        f.close();

        ct++;
    }
}

void QNPDSimulator::evaluateHessianPureConstraint(const PDVector& x, PDSparseMatrix& hessian_matrix)
{
    // spdlog::info(">>> QNPDSimulator::evaluateHessianPureConstraint()");
    hessian_matrix.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
    std::vector<PDSparseMatrixTriplet> h_triplets;
    h_triplets.clear();

    for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it) {
        (*it)->EvaluateHessian(x, h_triplets, m_definiteness_fix);
    }

    hessian_matrix.setFromTriplets(h_triplets.begin(), h_triplets.end());
    // spdlog::info(">>> QNPDSimulator::evaluateHessianPureConstraint() - After");

    if (m_processing_collision) {
        PDSparseMatrix HC;
        evaluateHessianCollision(x, HC);
        hessian_matrix += HC;
    }
}

void QNPDSimulator::evaluateHessianCollision(const PDVector& x, PDSparseMatrix& hessian_matrix)
{
    // spdlog::info(">>> QNPDSimulator::evaluateHessianCollision()");
    hessian_matrix.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
    std::vector<PDSparseMatrixTriplet> h_triplets;
    h_triplets.clear();

    for (std::vector<QNPDPenaltyConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it) {
        it->EvaluateHessian(x, h_triplets, m_definiteness_fix);
    }

    // spdlog::info(">>> QNPDSimulator::evaluateHessianCollision() - Before setFromTriplets");
    hessian_matrix.setFromTriplets(h_triplets.begin(), h_triplets.end());
    // spdlog::info(">>> QNPDSimulator::evaluateHessianCollision() - After");
}

PDScalar QNPDSimulator::evaluateEnergyPureConstraint(const PDVector& x, const PDVector& f_ext)
{
    PDScalar energy = 0.0;

    if (!m_enable_openmp) {
        for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it) {
            energy += (*it)->EvaluateEnergy(x);
        }
    }
    else {
        // openmp get all energy
        int i;
#pragma omp parallel
        {
#pragma omp for
            for (i = 0; i < m_constraints.size(); i++) {
                m_constraints[i]->EvaluateEnergy(x);
            }
        }

        // reduction
        for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it) {
            energy += (*it)->GetEnergy();
        }
    }

    // energy -= f_ext.dot(x);

    //// collision
    // for (unsigned int i = 1; i * 3 < x.size(); i++)
    //{
    //	EigenVector3 xi = x.block_vector(i);
    //	EigenVector3 n;
    //	PDScalar d;
    //	if (m_scene->StaticIntersectionTest(xi, n, d))
    //	{

    //	}
    //}

    // hardcoded collision plane
    if (m_processing_collision) {
        energy += evaluateEnergyCollision(x);
    }

    return energy;
}

PDScalar QNPDSimulator::evaluateEnergyCollision(const PDVector& x)
{
    PDScalar energy = 0.0;

    if (!m_enable_openmp) {
        for (std::vector<QNPDPenaltyConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it) {
            energy += it->EvaluateEnergy(x);
        }
    }
    else {
        // openmp get all energy
        int i;
#pragma omp parallel
        {
#pragma omp for
            for (i = 0; i < m_collision_constraints.size(); i++) {
                m_collision_constraints[i].EvaluateEnergy(x);
            }
        }

        // reduction
        for (std::vector<QNPDPenaltyConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it) {
            energy += it->GetEnergy();
        }
    }

    return energy;
}

PDScalar QNPDSimulator::evaluateEnergy(const PDVector& x)
{
    PDScalar energy_pure_constraints, energy;

    PDScalar inertia_term = 0.5 * (x - m_y).transpose() * m_mesh->m_mass_matrix * (x - m_y);
    PDScalar h_square = m_dt * m_dt;
    switch (m_integration_method) {
    case INTEGRATION_QUASI_STATICS:
        energy = evaluateEnergyPureConstraint(x, m_external_force);
        energy -= m_external_force.dot(x);
        break;
    case INTEGRATION_IMPLICIT_EULER:
        energy_pure_constraints = evaluateEnergyPureConstraint(x, m_external_force);
        energy = inertia_term + h_square * energy_pure_constraints;
        break;
    case INTEGRATION_IMPLICIT_BDF2:
        energy_pure_constraints = evaluateEnergyPureConstraint(x, m_external_force);
        energy = inertia_term + h_square * 4.0 / 9.0 * energy_pure_constraints;
        break;
    case INTEGRATION_IMPLICIT_MIDPOINT:
        energy_pure_constraints = evaluateEnergyPureConstraint((x + m_mesh->m_current_positions_vec) / 2, m_external_force);
        energy = inertia_term + h_square * (energy_pure_constraints);
        break;
    case INTEGRATION_IMPLICIT_NEWMARK_BETA:
        energy_pure_constraints = evaluateEnergyPureConstraint(x, m_external_force);
        energy = inertia_term + h_square / 4 * (energy_pure_constraints + m_z.dot(x));
        break;
    }

    return energy;
}

void QNPDSimulator::prefactorize()
{
    if (m_prefactorization_flag == false) {
        // update laplacian coefficients
        if (m_stiffness_auto_laplacian_stiffness) {
            m_stiffness_laplacian = m_constraints.at(0)->ComputeLaplacianWeight();
            m_precomputing_flag = false;
            m_prefactorization_flag = false;
            m_prefactorization_flag_newton = false;
        }
        else {
            setMaterialProperty(m_constraints);
        }

        // { // full space laplacian 3n x 3n
        //     // evaluateLaplacian(m_weighted_laplacian);
        //     evaluateLaplacianPureConstraint(m_weighted_laplacian);
        //     m_weighted_laplacian = m_mesh->m_mass_matrix + m_dt2 * m_weighted_laplacian;
        //     factorizeDirectSolverLLT(m_weighted_laplacian, m_prefactored_solver, "Our Method"); // prefactorization of laplacian
        //     m_preloaded_cg_solver.compute(m_weighted_laplacian); // load the cg solver
        // }

        { // reduced dim space laplacian n x n
            // evaluateLaplacian1D(m_weighted_laplacian_1D);
            evaluateLaplacianPureConstraint1D(m_weighted_laplacian_1D);
            m_weighted_laplacian_1D = m_mesh->m_mass_matrix + m_dt2 * m_weighted_laplacian_1D;
            factorizeDirectSolverLLT(m_weighted_laplacian_1D, m_prefactored_solver_1D, "QNPD Method Reduced Space");
            m_preloaded_cg_solver_1D.compute(m_weighted_laplacian_1D);
        }

        m_prefactorization_flag = true;
    }
}

void QNPDSimulator::factorizeDirectSolverLLT(const PDSparseMatrix& A, PDSparseLLTSolver& lltSolver, char* warning_msg)
{
    PDSparseMatrix A_prime = A;
    lltSolver.analyzePattern(A_prime);
    lltSolver.factorize(A_prime);
    PDScalar Regularization = 1e-10;
    bool success = true;
    PDSparseMatrix I;
    while (lltSolver.info() != Eigen::Success) {
        if (success == true) // first time factorization failed
        {
            // EigenMakeSparseIdentityMatrix(A.rows(), A.cols(), I);
            assert(A.rows() == A.cols());
            std::vector<PDSparseMatrixTriplet> triplets(A.rows());
            PD_PARALLEL_FOR
            for (int i = 0; i < A.rows(); i++) {
                triplets[i] = (PDSparseMatrixTriplet(i, i, 1));
            }
            I.resize(A.rows(), A.cols());
            I.setFromTriplets(triplets.begin(), triplets.end());
            I.makeCompressed();
        }
        Regularization *= 10;
        A_prime = A_prime + Regularization * I;
        lltSolver.factorize(A_prime);
        success = false;
    }
    if (!success && m_verbose_show_factorization_warning)
        std::cout << "Warning: " << warning_msg << " adding " << Regularization << " identites.(llt solver)" << std::endl;
}

PDScalar QNPDSimulator::linearSolve(PDVector& x, const PDSparseMatrix& A, const PDVector& b, char* msg)
{
    PDScalar residual = 0;

    switch (m_solver_type) {
    case SOLVER_TYPE_DIRECT_LLT: {
        PDSparseLLTSolver A_solver;
        factorizeDirectSolverLLT(A, A_solver, msg);
        x = A_solver.solve(b);
    } break;
    case SOLVER_TYPE_CG: {
        x.resize(b.size());
        x.setZero();
        residual = conjugateGradientWithInitialGuess(x, A, b, m_iterative_solver_max_iteration);
    } break;
    default:
        break;
    }

    return residual;
}

PDScalar QNPDSimulator::conjugateGradientWithInitialGuess(PDVector& x, const PDSparseMatrix& A, const PDVector& b, const unsigned int max_it /* = 200 */, const PDScalar tol /* = 1e-5 */)
{
    PDVector r = b - A * x;
    PDVector p = r;
    PDScalar rsold = r.dot(r);
    PDScalar rsnew;

    PDVector Ap;
    Ap.resize(x.size());
    PDScalar alpha;

    for (unsigned int i = 1; i != max_it; ++i) {
        Ap = A * p;
        alpha = rsold / p.dot(Ap);
        x = x + alpha * p;

        r = r - alpha * Ap;
        rsnew = r.dot(r);
        if (sqrt(rsnew) < tol) {
            break;
        }
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }

    return sqrt(rsnew);
}

void QNPDSimulator::LBFGSKernelLinearSolve(PDVector& r, PDVector rhs, PDScalar scaled_identity_constant) // Ar = rhs
{
    PROFILE_STEP("LBFGSKernelLinearSolve");
    r.resize(rhs.size());
    // LBFGS_H0_LAPLACIAN
    using EigenMatrixx3 = Eigen::Matrix<PDScalar, -1, 3, 0, -1, 3>;

    // solve the linear system in reduced dimension because of the pattern of the Laplacian matrix
    // convert to nx3 space
    EigenMatrixx3 rhs_n3(rhs.size() / 3, 3);
    { // Vector3mx1ToMatrixmx3(rhs, rhs_n3);
        PD_PARALLEL_FOR
        for (int i = 0; i < rhs_n3.rows(); i++) {
            rhs_n3.block<1, 3>(i, 0) = rhs.block<3, 1>(3 * i, 0).transpose();
        }
    }
    // solve using the nxn laplacian
    EigenMatrixx3 r_n3;
    if (m_solver_type == SOLVER_TYPE_CG) { // CG
        m_preloaded_cg_solver_1D.setMaxIterations(m_iterative_solver_max_iteration);
        r_n3 = m_preloaded_cg_solver_1D.solve(rhs_n3);
    }
    else { // Direct LLT
        r_n3 = m_prefactored_solver_1D.solve(rhs_n3);
    }
    // convert the result back
    { // Matrixmx3ToVector3mx1(r_n3, r);
        PD_PARALLEL_FOR
        for (int i = 0; i < r_n3.rows(); i++) {
            r.block<3, 1>(3 * i, 0) = r_n3.block<1, 3>(i, 0).transpose();
        }
    }

    ////// conventional solve using 3nx3n system
    // if (m_solver_type == SOLVER_TYPE_CG)
    //{
    //	m_preloaded_cg_solver.setMaxIterations(m_iterative_solver_max_iteration);
    //	r = m_preloaded_cg_solver.solve(rhs);
    // }
    // else
    //{
    //	r = m_prefactored_solver.solve(rhs);
    // }
}

PDScalar QNPDSimulator::lineSearch(const PDVector& x, const PDVector& gradient_dir, const PDVector& descent_dir)
{
    if (m_ls.enable_line_search) {
        PDVector x_plus_tdx(m_mesh->m_system_dimension);
        PDScalar t = 1.0 / m_ls.beta;
        // PDScalar t = m_ls_step_size/m_ls.beta;
        PDScalar lhs, rhs;

        PDScalar currentObjectiveValue;
        try {
            currentObjectiveValue = evaluateEnergy(x);
        }
        catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        }
        do {
#ifdef OUTPUT_LS_ITERATIONS
            g_total_ls_iterations++;
#endif
            t *= m_ls.beta;
            x_plus_tdx = x + t * descent_dir;

            lhs = 1e15;
            rhs = 0;
            try {
                lhs = evaluateEnergy(x_plus_tdx);
            }
            catch (const std::exception&) {
                continue;
            }
            rhs = currentObjectiveValue + m_ls.alpha * t * (gradient_dir.transpose() * descent_dir)(0);
            if (lhs >= rhs) {
                continue; // keep looping
            }

            break; // exit looping

        } while (t > 1e-5);

        if (t < 1e-5) {
            t = 0.0;
        }
        m_ls.step_size = t;

        if (m_verbose_show_converge) {
            std::cout << "Linesearch Stepsize = " << t << std::endl;
            std::cout << "lhs (current energy) = " << lhs << std::endl;
            std::cout << "previous energy = " << currentObjectiveValue << std::endl;
            std::cout << "rhs (previous energy + alpha * t * gradient.dot(descet_dir)) = " << rhs << std::endl;
        }

#ifdef OUTPUT_LS_ITERATIONS
        g_total_iterations++;
        if (g_total_iterations % OUTPUT_LS_ITERATIONS_EVERY_N_FRAMES == 0) {
            std::cout << "Avg LS Iterations = " << g_total_ls_iterations / g_total_iterations << std::endl;
            g_total_ls_iterations = 0;
            g_total_iterations = 0;
        }
#endif
        return t;
    }
    else {
        return m_ls.step_size;
    }
}

PDScalar QNPDSimulator::linesearchWithPrefetchedEnergyAndGradientComputing(const PDVector& x, const PDScalar current_energy, const PDVector& gradient_dir, const PDVector& descent_dir, PDScalar& next_energy, PDVector& next_gradient_dir)
{
    PROFILE_STEP("LINESEARCH");
    // TICK(LLINESEARCH_TIMECOST);
    if (!m_ls.enable_line_search) {
        return m_ls.step_size;
    }
    PDVector x_plus_tdx(m_mesh->m_system_dimension);
    PDScalar t = 1.0 / m_ls.beta;
    PDScalar lhs, rhs;
    PDScalar currentObjectiveValue = current_energy;

    do {
        t *= m_ls.beta;
        x_plus_tdx = x + t * descent_dir;

        lhs = 1e15;
        rhs = 0;
        try {
            lhs = evaluateEnergyAndGradient(x_plus_tdx, next_gradient_dir);
        }
        catch (const std::exception&) {
            continue;
        }
        rhs = currentObjectiveValue + m_ls.alpha * t * (gradient_dir.transpose() * descent_dir)(0);
        if (lhs >= rhs) {
            continue; // keep looping
        }

        next_energy = lhs;
        break; // exit looping
    } while (t > 1e-5);

    if (t < 1e-5) {
        t = 0.0;
        next_energy = current_energy;
        next_gradient_dir = gradient_dir;
    }
    m_ls.step_size = t;

    if (m_verbose_show_converge) {
        std::cout << "Linesearch Stepsize = " << t << std::endl;
        std::cout << "lhs (current energy) = " << lhs << std::endl;
        std::cout << "previous energy = " << currentObjectiveValue << std::endl;
        std::cout << "rhs (previous energy + alpha * t * gradient.dot(descet_dir)) = " << rhs << std::endl;
    }

    // TOCK(LLINESEARCH_TIMECOST);
    return t;
}

Eigen::MatrixXd QNPDSimulator::GetColorMapData()
{
    // if (m_colorMapType == 0) { // DISPLACEMENT
    //     return (m_mesh->m_positions - m_mesh->m_restpose_positions).cast<double>().rowwise().norm();
    // }
    static Eigen::MatrixXd m_E(m_mesh->m_positions.rows(), 1);
    if (m_frameCount % 5 == 0) {
        std::vector<PDScalar> tetsE(m_constraints.size());
        assert(m_constraints.size() == m_mesh->m_tets.rows());
        PD_PARALLEL_FOR
        for (int tInd = 0; tInd < m_constraints.size(); tInd++) {
            tetsE[tInd] = m_constraints[tInd]->EvaluateEnergy(m_mesh->m_current_positions_vec);
        }
        Eigen::MatrixXd _E(m_mesh->m_positions.rows(), 1);
        PD_PARALLEL_FOR
        for (int v = 0; v < m_mesh->m_positions.rows(); v++) {
            _E(v, 0) = 0;
            for (auto tp : m_mesh->m_tetsPerVertex[v]) {
                _E(v, 0) += tetsE[tp.first];
            }
            _E(v, 0) /= 4.0;
        }
        // if (isUniformColorMap) {
        if (true) {
            auto tmp = _E;
            PD_PARALLEL_FOR
            for (int v = 0; v < m_mesh->m_positions.rows(); v++) {
                _E.row(v).setZero();
                for (auto nv : m_mesh->m_adjVerts1rd[v]) {
                    _E.row(v) += tmp.row(nv);
                }
                _E.row(v) /= (m_mesh->m_adjVerts1rd[v].size() + 1);
            }
        }
        // spdlog::info(">>> QNPDSimulator::GetColorMapData()");
        return m_E = _E.rowwise().norm();
    }
}

}