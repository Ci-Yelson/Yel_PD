#include "NewtonSimulator.hpp"
#include <memory>

#include "polysolve/LinearSolver.hpp"

#include "UI/InteractState.hpp"
#include "Util/Profiler.hpp"

extern UI::InteractState g_InteractState;
extern Util::Profiler g_StepProfiler;

namespace PD {
namespace CL {

NewtonSimulator::NewtonSimulator(std::shared_ptr<TetMesh> tetMesh)
{
    // LoadParamsAndApply();
}

void NewtonSimulator::LoadParamsAndApply()
{
    m_dt = g_InteractState.timeStep;
    m_dt2 = m_dt * m_dt;
    m_dt2inv = 1.0 / m_dt2;

    // ...
}

void NewtonSimulator::PreCompute()
{
}

void NewtonSimulator::Step()
{
    PROFILE_STEP("HRPD_STEP");
    int _N = m_mesh->m_positions.rows();
    constexpr PDScalar NaN = std::numeric_limits<PDScalar>::quiet_NaN();
    Eigen::SparseMatrix<PDScalar> objHessian;
    // ---------------------------
    // Initialize the minimization
    // ---------------------------
    EvalFext();
    m_mesh->EvalGradient();
    Eigen::Matrix<PDScalar, -1, 1>& objGrad = m_mesh->m_Grad;
    Eigen::Matrix<PDScalar, -1, 1> delta_x;
    delta_x.setZero(_N * 3, 1);

    // Update inertia term - INTEGRATION_IMPLICIT_EULER
    Eigen::Matrix<PDScalar, -1, 1> s;
    s = m_mesh->m_positions_vec + m_dt * m_mesh->m_velocities_vec + m_dt2 * m_mesh->m_mass_matrix_inv * m_fExt;
    Eigen::Matrix<PDScalar, -1, 1> x = s;

    // Handle interaction - by directly edit `s` way.
    HandleInteraction(s);

    bool _CONVERGE = false;
    for (int i = 0; !_CONVERGE && i < 10; i++) {
        m_mesh->EvalGradient();
        m_mesh->EvalHessian();
        // Complete Gradient
        // objGrad = m_mesh->m_mass_matrix * (x - s) + m_dt2 * m_mesh->m_Grad;
        // Complete Hessian - INTEGRATION_IMPLICIT_EULER
        objHessian = m_mesh->m_mass_matrix + m_dt2 * m_mesh->m_H;

        // polysolve
        auto solver = polysolve::LinearSolver::create("Eigen::SimplicialLDLT", "");
        solver->analyzePattern(objHessian, objHessian.rows());
        solver->factorize(objHessian);
        solver->solve(m_mesh->m_Grad, delta_x);
        delta_x = -delta_x;

        // Line Search
        m_mesh->EvalEnergy();
        double alpha_k = Linesearch(x, m_mesh->m_E, m_mesh->m_Grad, delta_x, m_ls.prefetched_energy, m_ls.prefetched_gradient);
        x += alpha_k * delta_x;
    };

    m_mesh->m_velocities_vec = (x - m_mesh->m_positions_vec) / m_dt;
    m_mesh->m_positions_vec = x;
    m_mesh->full_to_reduced(m_mesh->m_positions, m_mesh->m_positions_vec);
}

void NewtonSimulator::EvalFext()
{
    m_fExt.setZero(m_mesh->m_system_dim);
    {
        // gravity
        PD_PARALLEL_FOR
        for (unsigned int i = 0; i < m_mesh->m_positions.rows(); ++i) {
            // m_fExt[3 * i + 1] += -m_gravity_constant;
        }
        // m_fExt *= m_mesh->m_mass_matrix;
    }
}

double NewtonSimulator::Linesearch(PDVector& x, PDScalar current_energy, PDVector& gradient_dir, PDVector& descent_dir, PDScalar& next_energy, PDVector& next_gradient_dir)
{
    if (!m_ls.enable_line_search) {
        return m_ls.step_size;
    }
    PDVector old_x = m_mesh->m_positions_vec;
    PDScalar t = 1.0 / m_ls.beta;
    PDScalar lhs, rhs;
    PDScalar currentObjectiveValue = current_energy;

    do {
        t *= m_ls.beta;
        m_mesh->m_positions_vec = x + t * descent_dir;

        lhs = 1e15;
        rhs = 0;
        try {
            m_mesh->EvalEnergy();
            lhs = m_mesh->m_E;
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
    m_mesh->m_positions_vec = old_x;

    if (false) {
        std::cout << "Linesearch Stepsize = " << t << std::endl;
        std::cout << "lhs (current energy) = " << lhs << std::endl;
        std::cout << "previous energy = " << currentObjectiveValue << std::endl;
        std::cout << "rhs (previous energy + alpha * t * gradient.dot(descet_dir)) = " << rhs << std::endl;
    }

    return t;
}

void NewtonSimulator::HandleInteraction(Eigen::Matrix<PDScalar, -1, 1>& s)
{
    // User interaction - Drag
    if (g_InteractState.draggingState.isDragging) {
        if (g_InteractState.draggingState.vertexUsed >= 0) {
            int v = g_InteractState.draggingState.vertexUsed;
            double dt2 = g_InteractState.timeStep * g_InteractState.timeStep;
            // s.row(v) += (1.0 / dt2) * g_InteractState.draggingState.force.cast<PDScalar>();
            auto add = (1.0 / dt2) * g_InteractState.draggingState.force.cast<PDScalar>();
            s[3 * v + 0] += add[0];
            s[3 * v + 1] += add[1];
            s[3 * v + 2] += add[2];
        }
    }
}

}
}