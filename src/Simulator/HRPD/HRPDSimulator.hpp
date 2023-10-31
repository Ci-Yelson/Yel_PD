/*
HRPD - Hyper-Reduced Projective Dynamic [2018] [TODO]
*/
#pragma once

#include "HRPDTetMesh.hpp"
#include "Simulator/HRPD/HRPDOperationManager.hpp"
#include "Simulator/HRPD/HRPDSubspaceBuilder.hpp"
#include "Simulator/PDTypeDef.hpp"
#include "Simulator/PDSimulator.hpp"
#include "UI/InteractState.hpp"
#include "Util/Timer.hpp"

#include <memory>
#include <vector>

namespace PD {

struct HRPDSimulator : public PDSimulator {
    std::shared_ptr<HRPDTetMesh> m_mesh;

    double m_stiffnessFactor = 1.0;

    PDPositions m_restPositionsSubspace; // [4k x 3]
    PDPositions m_positionsSubspace; // [4k x 3]
    PDPositions m_velocitiesSubspace; // [4k x 3]

    // ----------------- Config -----------------
    PDScalar m_dt{ 16 }, m_dt2{ 16 * 16 }, m_dt2inv{ 1.0f / (16 * 16) };

    // ----------------- Pre-computation -----------------
    SubspaceBuilder m_subspaceBuilder;
    // -- For global system
    PDMatrix m_subspaceLHS_mom; // UT * M * U [4k x 4k]
    PDMatrix m_subspaceLHS_inner; // UT * (ST * \lambda * S) * U [4k x 4k]
    // ---- Solve {LHSMatrix} * q_{sub} = {RHSMatrix}
    // PDDenseSolver m_subspaceSystemSolverDense;
    PDSparseSolver m_subspaceSystemSolverSparse;
    PDMatrix m_subspaceRHS_mom; // UT * M * U [4k x 4k]
    PDSparseMatrix m_subspaceRHSSparse_mom;

    // -- For f_{ext}
    PDPositions m_fExt;
    PDPositions m_fGravity;
    PDPositions m_fExtWeightedSubspace;
    PDPositions m_fGravityWeightedSubspace;

    // ----------------- Online-Step -----------------
    int m_frameCount{ 0 };
    bool m_collisionCorrection{ false };
    PDScalar m_frictionCoeff{ 0.5f };
    PDScalar m_repulsionCoeff{ 0.05f };
    PDScalar m_rayleighDampingAlpha{ 0 };
    // rhs0 - UT @ M/h^2 @ U @ s
    PDPositions rhs0;
    // rhs1 - UT @ ST @ WI @ V @ subP
    PDPositions rhs1;
    // rhs2 - (UT @ M/h^2 @ U @ s) + (UT @ ST @ WI @ V @ subP)
    PDPositions rhs2;
    // VTJTPused - VT * JT * P_{partial}
    PDPositions VTJTPused;
    // VPsub - V * P_{sub}
    PDPositions VPsub;
    PDPositions ms_s;
    PDPositions ms_prevPositionsSub;

public:
    // ----------------- UI-Operation -----------------
    // For Color Map
    int m_colorMapType = 0;
    bool isUniformColorMap = true;
    std::vector<std::string> ColorMapTypeStrs{
        "DISPLACEMENT",
        "INNER_FORCE_INTERPOL",
        "PD_ENERGY",
        "STVK_ENERGY",
        "PD_INNER_FORCE_FULL",
        "STVK_INNER_FORCE_FULL"
    };
    std::map<std::string, bool> ColorMapTypeActive{
        { "DISPLACEMENT", true },
        { "INNER_FORCE_INTERPOL", true },
        { "PD_ENERGY", true },
        { "STVK_ENERGY", true },
        { "PD_INNER_FORCE_FULL", false },
        { "STVK_INNER_FORCE_FULL", false }
    };
    PDPositions m_innerForceSubspace;
    PDPositions m_innerForceInterpol;
    PDMatrix m_PD_E;
    PDMatrix m_STVK_E;

public:
    HRPDSimulator(std::shared_ptr<HRPDTetMesh> tetMesh);

    void ComputeWeightedExtForces();
    void PreCompute();

    void LoadParamsAndApply() override;
    void Step() override;
    void Reset() override;

    void UpdateStiffnessWeight();
    void UpdateTimeStep() override;

    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) override
    {
        m_mesh->IGL_SetMesh(viewer, GetColorMapData());
        m_OpManager.IGL_SetMesh(viewer);
    }

    const PDPositions& GetRestPositions() override { return m_mesh->m_restpose_positions; }
    const PDPositions& GetPositions() override { return m_mesh->m_positions; }
    const PDTriangles& GetTriangles() override { return m_mesh->m_triangles; }

    // ------------------------------- Debug -------------------------------
    // bool m_debugStep = false;
    // void Debug_StoreData();

private:
    // ----------------- Online-Step -----------------
    void updateLHSMatrix();
    void handleGripAndCollisionsUsedVs(PDPositions& s, bool update);
    // void updateShowData(int showDataType);

public:
    // ----------------- API -----------------
    Eigen::MatrixXd GetColorMapData();
    void SetGravity(PDScalar gravityConstant);
    // void setFloor(PDScalar height, PDScalar floorCollisionWeight);
};
} // namespace PD