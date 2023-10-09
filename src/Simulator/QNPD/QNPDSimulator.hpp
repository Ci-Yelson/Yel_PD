// ## QNPD - Quasi-Newton [2017]
#pragma once

#include "QNPDConstraint.hpp"
#include "QNPDTetMesh.hpp"
#include "Simulator/PDSimulator.hpp"
#include "Simulator/PDTypeDef.hpp"

#include <deque>
#include <memory>

#define EPSILON 1e-6
#define EPSILON_SQUARE 1e-12
#define LARGER_EPSILON 1e-4

namespace PD {

typedef enum {
    INTEGRATION_IMPLICIT_EULER,
    INTEGRATION_IMPLICIT_BDF2,
    INTEGRATION_IMPLICIT_MIDPOINT,
    INTEGRATION_IMPLICIT_NEWMARK_BETA,
    INTEGRATION_QUASI_STATICS,
    INTEGRATION_TOTAL_NUM

} IntegrationMethod;

typedef enum {
    OPTIMIZATION_METHOD_GRADIENT_DESCENT,
    OPTIMIZATION_METHOD_NEWTON,
    OPTIMIZATION_METHOD_LBFGS,
    OPTIMIZATION_METHOD_TOTAL_NUM

} OptimizationMethod;

typedef enum {
    SOLVER_TYPE_DIRECT_LLT,
    SOLVER_TYPE_CG,
    SOLVER_TYPE_TOTAL_NUM
} SolverType;

struct QNPDSimulator : public PDSimulator {
    std::shared_ptr<QNPDTetMesh> m_mesh;
    std::vector<Constraint*> m_constraints;
    // collision constraints
    bool m_processing_collision{ true };
    std::vector<QNPDPenaltyConstraint> m_collision_constraints;

    // Simulation states
    PDVector m_qn; // 1x3m
    PDVector m_vn; // 1x3m

    // constant term in optimization:
    // 0.5(x-y)^2 M (x-y) + (c) * h^2 * E(x) - h^2 * x^T * z;
    PDVector m_y;
    PDVector m_z;

    // external force (gravity, wind, etc...)
    PDVector m_external_force;

    PDSparseMatrix m_weighted_laplacian_1D;
    // PDSparseMatrix m_weighted_laplacian;

    // for prefactorization
    bool m_precomputing_flag;
    bool m_prefactorization_flag;
    bool m_prefactorization_flag_newton;

#ifdef PARDISO_SUPPORT
    Eigen::PardisoLLT<PDSparseMatrix, Eigen::Upper> m_prefactored_solver;
    Eigen::PardisoLLT<PDSparseMatrix, Eigen::Upper> m_newton_solver;
    Eigen::PardisoLLT<PDSparseMatrix, Eigen::Upper> m_prefactored_solver_restpose_hessian;
    Eigen::PardisoLLT<PDSparseMatrix, Eigen::Upper> m_prefactored_solver_dual;
#else
    Eigen::SimplicialLLT<PDSparseMatrix, Eigen::Upper> m_prefactored_solver_1D;
    // Eigen::SimplicialLLT<PDSparseMatrix, Eigen::Upper> m_prefactored_solver;
    Eigen::SimplicialLLT<PDSparseMatrix, Eigen::Upper> m_newton_solver;
#endif
    Eigen::ConjugateGradient<PDSparseMatrix> m_preloaded_cg_solver_1D;
    // Eigen::ConjugateGradient<PDSparseMatrix> m_preloaded_cg_solver;

    struct LBFGS {
        bool is_first_iteration{ true };
        bool restart_every_frame{ false };
        bool need_update_H0{ true };
        int m{ 5 }, max_m{ 5 };
        PDVector last_x, last_gradient;
        std::deque<PDVector> s_queue, y_queue;
    } m_lbfgs;

    struct LS { // line search
        bool enable_line_search{ true };
        bool enable_exact_search{ false };
        PDScalar alpha{ 0.03 }, beta{ 0.5 };
        PDScalar step_size{ 1.0 };
        // prefetched instructions in linesearch
        bool is_first_iteration;
        PDVector prefetched_gradient;
        PDScalar prefetched_energy;
    } m_ls;

    // ----------------- Config -----------------
    PDScalar m_dt{ 0.0333 }; // 1/30 second.
    PDScalar m_dt2 = m_dt * m_dt;
    PDScalar m_dt2inv = 1.0 / m_dt2;
    // floor
    PDScalar m_floorCollisionWeight{ -1 };
    PDScalar m_floorHeight{ 0 };
    PDScalar m_damping_coefficient{ 0.001 };

    IntegrationMethod m_integration_method = INTEGRATION_IMPLICIT_EULER;
    OptimizationMethod m_optimization_method = OPTIMIZATION_METHOD_LBFGS;
    SolverType m_solver_type{ SOLVER_TYPE_DIRECT_LLT };
    int m_iterative_solver_max_iteration{ 10 };

    // for Newton's method
    bool m_definiteness_fix;

    MaterialType m_material_type{ MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG };
    PDScalar m_stiffness_attachment{ 120 };
    PDScalar m_stiffness_stretch{ 80 };
    // PDScalar m_stiffness_high{20};
    PDScalar m_stiffness_bending{ 20 };
    PDScalar m_stiffness_kappa{ 100 };
    PDScalar m_stiffness_laplacian{ 2 * m_stiffness_stretch + m_stiffness_bending };
    bool m_stiffness_auto_laplacian_stiffness{ true };

    int m_current_iteration{ 0 };

    bool m_verbose_show_converge{ false };
    bool m_verbose_show_factorization_warning{ true };
    bool m_enable_openmp{ true };

public:
    QNPDSimulator(std::shared_ptr<QNPDTetMesh> tetMesh);
    ~QNPDSimulator();

    void LoadParamsAndApply() override;

    void Step() override;
    void Reset() override;

    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) override { m_mesh->IGL_SetMesh(viewer); }

    const PDPositions& GetRestPositions() override { return m_mesh->m_restpose_positions; }
    const PDPositions& GetPositions() override { return m_mesh->m_positions; }
    const PDTriangles& GetTriangles() override { return m_mesh->m_triangles; }

    void UpdateTimeStep() override;

private:
    void setMaterialProperty(std::vector<Constraint*>& constraints);
    void setupConstraints(); // initialize constraints
    void clearConstraints(); // cleanup all constraints

    void calculateExternalForce();

    void collisionDetection(const PDVector& x);

    bool performGradientDescentOneIteration(PDVector& x);
    bool performNewtonsMethodOneIteration(PDVector& x);
    bool performLBFGSOneIteration(PDVector& x);

    void evaluateGradient(const PDVector& x, PDVector& gradient, bool enable_omp = true);
    void evaluateGradientPureConstraint(const PDVector& x, const PDVector& f_ext, PDVector& gradient);
    void evaluateGradientCollision(const PDVector& x, PDVector& gradient);
    PDScalar evaluateEnergyAndGradient(const PDVector& x, PDVector& gradient);
    PDScalar evaluateEnergyAndGradientPureConstraint(const PDVector& x, PDVector& gradient);
    PDScalar evaluateEnergyAndGradientCollision(const PDVector& x, PDVector& gradient);
    void evaluateHessian(const PDVector& x, PDSparseMatrix& hessian_matrix);
    void evaluateHessianPureConstraint(const PDVector& x, PDSparseMatrix& hessian_matrix);
    void evaluateHessianCollision(const PDVector& x, PDSparseMatrix& hessian_matrix);
    void evaluateLaplacianPureConstraint(PDSparseMatrix& laplacian_matrix);
    void evaluateLaplacianPureConstraint1D(PDSparseMatrix& laplacian_matrix_1d);

    PDScalar evaluateEnergyPureConstraint(const PDVector& x, const PDVector& f_ext);
    PDScalar evaluateEnergy(const PDVector& x);
    PDScalar evaluateEnergyCollision(const PDVector& x);

    PDScalar conjugateGradientWithInitialGuess(PDVector& x, const PDSparseMatrix& A, const PDVector& b, const unsigned int max_it = 200, const PDScalar tol = 1e-5);

    void prefactorize();
    void factorizeDirectSolverLLT(const PDSparseMatrix& A, PDSparseLLTSolver& lltSolver, char* warning_msg);

    PDScalar linearSolve(PDVector& x, const PDSparseMatrix& A, const PDVector& b, char* msg = "");
    void LBFGSKernelLinearSolve(PDVector& r, PDVector rhs, PDScalar scaled_identity_constant); // Ar = rhs
    PDScalar lineSearch(const PDVector& x, const PDVector& gradient_dir, const PDVector& descent_dir);
    PDScalar linesearchWithPrefetchedEnergyAndGradientComputing(const PDVector& x, const PDScalar current_energy, const PDVector& gradient_dir, const PDVector& descent_dir, PDScalar& next_energy, PDVector& next_gradient_dir);
};

}