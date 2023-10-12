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
    PDScalar m_dt{ 0.0333 }; // 1/30 second.
    PDScalar m_dt2 = m_dt * m_dt;
    PDScalar m_dt2inv = 1.0 / m_dt2;
    void UpdateTimeStep() override
    {
        // TODO
    }

public: // Simulation data
    PDVector m_s;

    // -- For f_{ext}
    PDVector m_fExt_vec;
    void EvalFext();

    PDScalar m_objE;
    void EvalObjEnergy(const PDVector& x);

public: // line search
    struct LineSearchConfig {
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

    Eigen::MatrixXd GetColorMapData()
    { // TODO
        return Eigen::MatrixXd(1, 1);
    }

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
