#pragma once

#include "TetMesh.hpp"

#include "Simulator/PDSimulator.hpp"
#include "Simulator/PDTypeDef.hpp"
#include <memory>

namespace PD {

namespace CL {

struct NewtonSimulator : public PDSimulator {
    std::shared_ptr<TetMesh> m_mesh;

    // ----------------- Config -----------------
    // TODO: USE GLOBAL dt ?
    PDScalar m_dt{ 16 }, m_dt2{ 16 * 16 }, m_dt2inv{ 1.0f / (16 * 16) };
    void UpdateTimeStep() override;

    // Simulation data
    // -- For f_{ext}
    PDPositions m_fExt;
    void EvalFext();

public:
    struct LineSearchConfig { // line search
        bool enable_line_search;
        bool enable_exact_search;
        PDScalar alpha, beta;
        PDScalar step_size;
        // prefetched instructions in linesearch
        bool is_first_iteration;
        PDVector prefetched_gradient;
        PDScalar prefetched_energy;
    } m_ls;
    double Linesearch(PDVector& x, PDScalar current_energy, PDVector& gradient_dir, PDVector& descent_dir, PDScalar& next_energy, PDVector& next_gradient_dir);

public:
    void HandleInteraction(Eigen::Matrix<PDScalar, -1, 1>& s);

public:
    NewtonSimulator(std::shared_ptr<TetMesh> tetMesh);

    Eigen::MatrixXd GetColorMapData(); //todo

    void PreCompute();
    void LoadParamsAndApply() override;
    void Step() override;
    void Reset() override;

    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) override
    {
        m_mesh->IGL_SetMesh(viewer, GetColorMapData());
        // m_OpManager.IGL_SetMesh(viewer);
    }

    const PDPositions& GetRestPositions() override { return m_mesh->m_restpose_positions; }
    const PDPositions& GetPositions() override { return m_mesh->m_positions; }
    const PDTriangles& GetTriangles() override { return m_mesh->m_triangles; }
};

}

}
