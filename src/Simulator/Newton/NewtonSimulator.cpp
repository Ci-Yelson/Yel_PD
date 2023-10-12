#include "NewtonSimulator.hpp"
#include <fstream>
#include <memory>
#include <string>

// TODO
// #include "polysolve/LinearSolver.hpp"

#include "Simulator/Newton/NewtonSimulator.hpp"
#include "Simulator/PDTypeDef.hpp"
#include "UI/InteractState.hpp"
#include "Util/Profiler.hpp"
#include "Util/Timer.hpp"

extern UI::InteractState g_InteractState;
extern Util::Profiler g_StepProfiler;

namespace PD {
namespace CL {

NewtonSimulator::NewtonSimulator(std::shared_ptr<TetMesh> tetMesh)
    : m_mesh(tetMesh)
{
    LoadParamsAndApply();
}

void NewtonSimulator::LoadParamsAndApply()
{
    m_dt = g_InteractState.timeStep;
    m_dt2 = m_dt * m_dt;
    m_dt2inv = 1.0 / m_dt2;

    {
        m_ls.enable_line_search = g_InteractState.newtonParams.ls_enable_line_search;
        m_ls.enable_exact_search = g_InteractState.newtonParams.ls_enable_exact_search;
        m_ls.step_size = g_InteractState.newtonParams.ls_step_size;
        m_ls.alpha = g_InteractState.newtonParams.ls_alpha;
        m_ls.beta = g_InteractState.newtonParams.ls_beta;
    }

    // ...
    PreCompute();
}

void NewtonSimulator::PreCompute()
{
    PROFILE_PREC("Newton_PRECOMPUTE");
    spdlog::info("NewtonSimulator::PreCompute()");

    {
        PROFILE_PREC("0_TETCONSTRAINTS");
        TICKC(HRPD_PRECOMPUTE__TETCONSTRAINTS);
        spdlog::info("> HRPDSimulator::PreCompute() - 0_TETCONSTRAINTS");
        m_mesh->InitTetConstraints(
            g_InteractState.newtonParams.stifness_mu, g_InteractState.newtonParams.stifness_lambda);
        TOCKC(HRPD_PRECOMPUTE__TETCONSTRAINTS);
    }
}

void NewtonSimulator::Step()
{
    PROFILE_STEP("NEWTON_STEP");
    int _N = m_mesh->m_positions.rows();
    constexpr PDScalar NaN = std::numeric_limits<PDScalar>::quiet_NaN();
    // ---------------------------
    // Initialize the minimization
    // ---------------------------
    PDVector x_pre = m_mesh->m_positions_vec;
    EvalFext();

    // Update inertia term - INTEGRATION_IMPLICIT_EULER
    m_s = m_mesh->m_positions_vec + m_dt * m_mesh->m_velocities_vec + m_dt2 * m_fExt_vec;

    // Handle interaction - by directly edit `s` way.
    HandleInteraction(m_s);

    static int timestep = 0;
    {
        std::ifstream f;
        f.open("./debug/QNPD/m_y/" + std::to_string(timestep));
        for (int i = 0; i < m_s.rows(); i++) {
            f >> m_s(i);
        }
        f.close();
    }
    {
        std::ifstream f;
        f.open("./debug/QNPD/objs/" + std::to_string(timestep) + ".obj");
        char ch;
        for (int i = 0; i < m_mesh->m_positions.rows(); i++) {
            f >> ch;
            f >> m_mesh->m_positions(i, 0) >> m_mesh->m_positions(i, 1) >> m_mesh->m_positions(i, 2);
        }
        f.close();
        m_mesh->reduced_to_full(m_mesh->m_positions, m_mesh->m_positions_vec);
        m_mesh->m_velocities_vec = (m_mesh->m_positions_vec - x_pre) / m_dt;
    }

    PDVector x = m_s;

    EvalObjEnergy(x);
    Eigen::Matrix<PDScalar, -1, 1> objGrad(m_mesh->m_Grad.rows(), m_mesh->m_Grad.cols());
    Eigen::SparseMatrix<PDScalar> objHessian;
    Eigen::Matrix<PDScalar, -1, 1> delta_x;
    delta_x.setZero(_N * 3, 1);

    bool _CONVERGE = false;
    for (int i = 0; !_CONVERGE && i < 10; i++) {
        spdlog::info(">>> NewtonSimulator::Step() - Substep = {}", i);
        // Complete Gradient
        m_mesh->EvalGradient(x);
        objGrad = m_mesh->m_mass_matrix * (x - m_s) + m_dt2 * m_mesh->m_Grad;
        {
            static int ct = 0;
            if (ct < 100) {
                std::ofstream f;
                f.open("./debug/NT/grad_pure/" + std::to_string(ct));
                f << fmt::format("Size = ({}, {})\n", m_mesh->m_Grad.rows(), m_mesh->m_Grad.cols());
                f << m_mesh->m_Grad;
                f.close();

                f.open("./debug/NT/grad/" + std::to_string(ct));
                f << fmt::format("Size = ({}, {})\n", objGrad.rows(), objGrad.cols());
                f << objGrad;
                f.close();

                ct++;
            }
        }
        // Complete Hessian - INTEGRATION_IMPLICIT_EULER
        m_mesh->EvalHessian(x);
        objHessian = m_mesh->m_mass_matrix + m_dt2 * m_mesh->m_H;
        {
            static int ct = 0;
            if (ct < 100) {
                std::ofstream f;
                f.open("./debug/NT/hessian_pure/" + std::to_string(ct));
                f << fmt::format("Size = ({}, {})\n", m_mesh->m_H.rows(), m_mesh->m_H.cols());
                f << m_mesh->m_H;
                f.close();

                f.open("./debug/NT/hessian/" + std::to_string(ct));
                f << fmt::format("Size = ({}, {})\n", objHessian.rows(), objHessian.cols());
                f << objHessian;
                f.close();

                ct++;
            }
        }

        {
            // TODO: polysolve
            // auto solver = polysolve::LinearSolver::create("Eigen::SimplicialLDLT", "");
            // solver->analyzePattern(objHessian, objHessian.rows());
            // solver->factorize(objHessian);
            // solver->solve(m_mesh->m_Grad, delta_x);
            // delta_x = -delta_x;
        } { // Debug
            objHessian.setIdentity();
        }
        PDSparseSolver solver;
        solver.analyzePattern(objHessian);
        solver.factorize(objHessian);
        // delta_x = solver.solve(m_mesh->m_Grad);
        delta_x = solver.solve(objGrad);
        delta_x = -delta_x;

        // Line Search
        EvalObjEnergy(x);
        spdlog::info(">>> Before LS - m_objE = {}", m_objE);
        double alpha_k = Linesearch(x, m_objE, objGrad, delta_x, m_ls.prefetched_energy, m_ls.prefetched_gradient);
        x += alpha_k * delta_x;
        m_mesh->m_positions_vec = x;

        PDScalar gradNorm = delta_x.norm();
        spdlog::info(">>> NewtonSimulator::Step() - Substep = {}, grad_norm = {}", i, gradNorm);
    };

    m_mesh->m_velocities_vec = (x - x_pre) / m_dt;
    m_mesh->m_positions_vec = x;
    m_mesh->full_to_reduced(m_mesh->m_positions, m_mesh->m_positions_vec);

    {
        std::ofstream f;
        f.open("./debug/QNPD/m_y/" + std::to_string(timestep));
        f << m_s;
        f.close();
    }
}

void NewtonSimulator::Reset()
{
    { // reset pos, vel, ...
        m_mesh->ResetPositions();
    }
}

void NewtonSimulator::EvalFext()
{
    m_fExt_vec.setZero(m_mesh->m_system_dim);
    {
        PD_PARALLEL_FOR
        for (int i = 0; i < m_mesh->m_positions.rows(); ++i) {
            m_fExt_vec[3 * i + 1] += -g_InteractState.newtonParams.gravityConstant * m_mesh->m_vertexMasses[i];
            for (int j = 0; j < 3; j++) {
                m_fExt_vec[3 * i + j] /= m_mesh->m_vertexMasses[i];
            }
        }
    }
}

void NewtonSimulator::EvalObjEnergy(const PDVector& x)
{
    PDScalar inertia_term = 0.5 * (x - m_s).transpose() * m_mesh->m_mass_matrix * (x - m_s);
    m_mesh->EvalEnergy(x);
    m_objE = inertia_term + m_dt2 * m_mesh->m_E;
}

double NewtonSimulator::Linesearch(PDVector& x, PDScalar current_energy, PDVector& gradient_dir, PDVector& descent_dir, PDScalar& next_energy, PDVector& next_gradient_dir)
{
    spdlog::info(">>> NewtonSimulator::Linesearch - Enable = {}", m_ls.enable_line_search);
    if (!m_ls.enable_line_search) {
        return m_ls.step_size;
    }
    PDVector x_plus_tdx(m_mesh->m_system_dim);
    PDScalar t = 1.0 / m_ls.beta;
    PDScalar lhs, rhs;
    PDScalar currentObjectiveValue = current_energy;

    int iter_cnt = 0;
    do {
        iter_cnt++;
        t *= m_ls.beta;
        x_plus_tdx = x + t * descent_dir;

        lhs = 1e15;
        rhs = 0;
        try {
            EvalObjEnergy(x_plus_tdx);
            lhs = m_objE;
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

    { // log
        spdlog::info("Linesearch Iters = {}, Stepsize = {}, E_curr = {}, E_prev = {}", iter_cnt, t, lhs, currentObjectiveValue);
    }

    // if (false) {
    //     std::cout << "Linesearch Stepsize = " << t << std::endl;
    //     std::cout << "lhs (current energy) = " << lhs << std::endl;
    //     std::cout << "previous energy = " << currentObjectiveValue << std::endl;
    //     std::cout << "rhs (previous energy + alpha * t * gradient.dot(descet_dir)) = " << rhs << std::endl;
    // }

    return t;
}

void NewtonSimulator::HandleInteraction(Eigen::Matrix<PDScalar, -1, 1>& s)
{
    // User interaction - Drag
    if (g_InteractState.draggingState.isDragging) {
        if (g_InteractState.draggingState.vertex >= 0) {
            int v = g_InteractState.draggingState.vertex;
            double dt2 = g_InteractState.timeStep * g_InteractState.timeStep;
            // s.row(v) += (1.0 / dt2) * g_InteractState.draggingState.force.cast<PDScalar>();
            auto add = (1.0 / dt2) * g_InteractState.draggingState.force.cast<PDScalar>();
            s[3 * v + 0] += add[0];
            s[3 * v + 1] += add[1];
            s[3 * v + 2] += add[2];
        }
    }
    PD_PARALLEL_FOR
    for (int i = 1; i < s.rows(); i += 3) {
        if (s[i] < g_InteractState.floorHeight) {
            s[i] = g_InteractState.floorHeight;
        }
    }
}

}
}