
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Simulator/HRPD/HRPDSimulator.hpp"
#include "Simulator/HRPD/HRPDSubspaceBuilder.hpp"
#include "Simulator/PDTypeDef.hpp"

#include "UI/InteractState.hpp"
#include "Util/Profiler.hpp"
#include "Util/StoreData.hpp"
#include "Util/Timer.hpp"
// #include "Util/MathHelper.hpp"

#include "spdlog/spdlog.h"

extern UI::InteractState g_InteractState;
extern Util::Profiler g_StepProfiler;
extern Util::Profiler g_PreComputeProfiler;

namespace PD {

HRPDSimulator::HRPDSimulator(std::shared_ptr<HRPDTetMesh> tetMesh)
    : m_mesh(tetMesh)
{
    m_fExt.setZero(m_mesh->m_positions.rows(), 3);
    m_fGravity.setZero(m_mesh->m_positions.rows(), 3);

    LoadParamsAndApply();

    if (g_InteractState.isGravityActive) SetGravity(g_InteractState.hrpdParams.gravityConstant);
}

void HRPDSimulator::LoadParamsAndApply()
{
    m_dt = g_InteractState.timeStep;
    m_dt2 = m_dt * m_dt;
    m_dt2inv = 1.0 / m_dt2;

    TICKC(HRPD_PRECOMPUTE__TOTAL);
    PreCompute();
    TOCKC(HRPD_PRECOMPUTE__TOTAL);
}

void HRPDSimulator::PreCompute()
{
    PROFILE_PREC("HRPD_PRECOMPUTE");
    spdlog::info("HRPDSimulator::PreCompute()");

    {
        PROFILE_PREC("0_TETCONSTRAINTS");
        TICKC(HRPD_PRECOMPUTE__TETCONSTRAINTS);
        spdlog::info("> HRPDSimulator::PreCompute() - 0_TETCONSTRAINTS");
        m_mesh->InitTetConstraints(
            g_InteractState.hrpdParams.wi,
            g_InteractState.hrpdParams.sigmaMax,
            g_InteractState.hrpdParams.sigmaMin);
        TOCKC(HRPD_PRECOMPUTE__TETCONSTRAINTS);
    }
    m_subspaceBuilder.Init(m_mesh);
    auto& _sub = m_subspaceBuilder;
    {
        PROFILE_PREC("1_POS-SUBSPACE");
        TICKC(HRPD_PRECOMPUTE__POS_SUBSPACE);
        spdlog::info("> HRPDSimulator::PreCompute() - 1_POS-SUBSPACE");
        _sub.CreateSubspacePosition();
        _sub.InitProjection();
        _sub.ProjectFullspaceToSubspaceForPos(m_positionsSubspace, m_mesh->m_positions);
        _sub.ProjectFullspaceToSubspaceForPos(m_velocitiesSubspace, m_mesh->m_velocities);
        {
            // check if the projection is correct
            m_mesh->m_positions = _sub.m_U * m_positionsSubspace;
            if (m_mesh->m_positions.hasNaN()) {
                spdlog::error("Warning: subspace projection did not work!");
            }
        }
        m_restPositionsSubspace = m_positionsSubspace;
        ComputeWeightedExtForces();
        TOCKC(HRPD_PRECOMPUTE__POS_SUBSPACE);
    }
    {
        PROFILE_PREC("2_PROJ-SUBSPACE");
        TICKC(HRPD_PRECOMPUTE__PROJ_SUBSPACE);
        spdlog::info("> HRPDSimulator::PreCompute() - 2_PROJ-SUBSPACE");
        _sub.CreateSubspaceProjection();
        _sub.CreateProjectionInterpolationMatrix();
        _sub.InitInterpolation();
        TOCKC(HRPD_PRECOMPUTE__PROJ_SUBSPACE);
    }
    { // Prepare for global system
        PROFILE_PREC("3_GLOBAL-LHS");
        TICKC(HRPD_PRECOMPUTE__GLOBAL_LHS);
        spdlog::info("> HRPDSimulator::PreCompute() - 3_GLOBAL-LHS");
        PDMatrix& UTMU = _sub.m_UTMU;
        PDMatrix eps(UTMU.rows(), UTMU.rows());
        eps.setIdentity();
        eps *= 1e-10;
        m_subspaceLHS_mom = UTMU + eps; // ! eps for sparse solver stability
        m_subspaceRHS_mom = UTMU;
        m_subspaceRHSSparse_mom = m_subspaceRHS_mom.sparseView(0, PD_SPARSITY_CUTOFF);

        m_subspaceLHS_inner.setZero(UTMU.rows(), UTMU.rows());
        m_subspaceLHS_inner += (_sub.m_UT * (_sub.m_ST * _sub.m_weightMatrixForProj * _sub.m_ST.transpose()) * _sub.m_U);
        PDMatrix lhsMatrix = m_subspaceLHS_mom * m_dt2inv + (g_InteractState.hrpdParams.wi / m_mesh->m_stiffness_weight) * m_subspaceLHS_inner;
        PDSparseMatrix lhsMatrixSampledSparse = lhsMatrix.sparseView(0, PD_SPARSITY_CUTOFF);
        m_subspaceSystemSolverSparse.compute(lhsMatrixSampledSparse);
        if (m_subspaceSystemSolverSparse.info() != Eigen::Success) {
            spdlog::warn("Warning: Factorization of the sparse LHS matrix for the global step was not successful!");
            PDSparseMatrix eps(lhsMatrixSampledSparse.rows(), lhsMatrixSampledSparse.rows());
            eps.setIdentity();
            eps *= 1e-12;
            while (m_subspaceSystemSolverSparse.info() != Eigen::Success && eps.coeff(0, 0) < 1e-10) {
                spdlog::warn("Adding small diagonal entries ({})...", eps.coeff(0, 0));
                lhsMatrixSampledSparse += eps;
                eps *= 2;
                m_subspaceSystemSolverSparse.compute(eps);
            }
        }
        TOCKC(HRPD_PRECOMPUTE__GLOBAL_LHS);
    }
}

void HRPDSimulator::Step()
{
    PROFILE_STEP("HRPD_STEP");
    PDPositions s;
    PDPositions oldPosSub = m_positionsSubspace;
    PDPositions tempAuxils;
    PDPositions tempSolution;
    bool _debug = false;

    auto& _sub = m_subspaceBuilder;
    // 1. Get inertia s
    {
        PROFILE_STEP("GET_INERTIA_s");
        if (_debug) spdlog::info("> HRPDSimulator::Step() - GET_INERTIA_s");
        if (m_collisionCorrection) {
            // Get actual velocities for used vertices
            _sub.InterpolateSubspaceToFullspafceForPos(_sub.m_velocitiesUsedVs, m_velocitiesSubspace, true);
            // Remove tangential movement and add repulsion movement from collided vertices
            PD_PARALLEL_FOR
            for (int v = 0; v < _sub.m_velocitiesUsedVs.rows(); v++) {
                if (_sub.m_velocitiesUsedVsRepulsion.row(v).norm() > 1e-12) {
                    PD3dVector curV = _sub.m_velocitiesUsedVs.row(v);
                    PD3dVector corrV = _sub.m_velocitiesUsedVsRepulsion.row(v);
                    PDVector tangentialV = curV - curV.dot(corrV) * corrV;
                    tangentialV *= (1.0 - m_frictionCoeff);
                    tangentialV += corrV * m_repulsionCoeff;
                    _sub.m_velocitiesUsedVs.row(v) = tangentialV;
                }
            }
            // Project back to subspace
            PD_PARALLEL_FOR
            for (int d = 0; d < 3; d++) {
                m_velocitiesSubspace.col(d) = _sub.m_usedSubspaceSolverSparse.solve(_sub.m_UT_used * _sub.m_velocitiesUsedVs.col(d));
            }
            m_collisionCorrection = false;
        }
        s = m_positionsSubspace + m_dt * m_velocitiesSubspace + m_dt2 * (m_fExtWeightedSubspace + m_fGravityWeightedSubspace);
        if (m_rayleighDampingAlpha > 0) {
            s -= m_dt * m_rayleighDampingAlpha * m_velocitiesSubspace;
        }
        m_positionsSubspace = s;
        // Get actual positions for used vertices
        _sub.InterpolateSubspaceToFullspafceForPos(_sub.m_positionsUsedVs, m_positionsSubspace, true);
        // Correct vertex positions of gripped and collided vertices
        handleGripAndCollisionsUsedVs(s, true);
    }

    // rhs0 - (UT @ M @ U) /h^2 @ s
    rhs0 = m_subspaceRHSSparse_mom * m_dt2inv * s;
    // 2. Local / global loop
    for (int i = 0; i < g_InteractState.numIterations; i++) {
        { // 2.1. Local step: Constraint projections
            PROFILE_STEP("LOCAL_ITER");
            if (_debug) spdlog::info("> HRPDSimulator::Step() - LOCAL_ITER [{}]", std::to_string(i));
            if (i != 0) { // CollisionHandle
                // Get actual positions for used vertices
                _sub.InterpolateSubspaceToFullspafceForPos(_sub.m_positionsUsedVs, m_positionsSubspace, true);
                // Correct vertex positions of gripped and collided vertices
                // handleGripAndCollisionsUsedVs(m_positionsSubspace, true);
            }

            // rhs2 - (UT @ M/h^2 @ U @ s) + (UT @ ST @ WI @ V @ subP)
            // TICK(LOCAL_STEP_TC);
            rhs2 = rhs0;
            // for (auto& sub : m_subspaceBuilders) {
            {
                auto& sub = m_subspaceBuilder;
                tempAuxils.resize(3 * sub.m_sampledConstraintInds.size(), 3);
                tempSolution.resize(sub.m_V.cols(), 3);
                // Collect auxiliary variables for sampled constraints
                PD_PARALLEL_FOR
                for (int i = 0; i < sub.m_sampledConstraintInds.size(); i++) {
                    tempAuxils.block(i * 3, 0, 3, 3) = m_mesh->GetP(sub.m_sampledConstraintInds[i]);
                }
                // Set up rhs from auxiliaries and solve the system for all three columns
                PD_PARALLEL_FOR
                for (int d = 0; d < 3; d++) {
                    // VT @ JT @ P_{partial}
                    tempSolution.col(d) = sub.m_fittingSolver.solve(sub.ms_VTJT * tempAuxils.col(d));

                    // Apply finalization matrix and add to current rhs
                    rhs2.col(d) += (g_InteractState.hrpdParams.wi / m_mesh->m_stiffness_weight) * sub.ms_UTSTWiV * tempSolution.col(d);
                }
            }
            // TOCK(LOCAL_STEP_TC);
        }

        { // 2.2. Global step: Solve the linear system with fixed constrain projections
            PROFILE_STEP("GLOBAL_ITER");
            if (_debug) spdlog::info("> HRPDSimulator::Step() - GLOBAL_ITER [{}]", std::to_string(i));
            // TICK(GLOBAL_STEP_TC);
            // Solve, for x, y and z in parallel
            PD_PARALLEL_FOR
            for (int d = 0; d < 3; d++) {
                m_positionsSubspace.col(d) = m_subspaceSystemSolverSparse.solve(rhs2.col(d));
            }
            // TOCK(GLOBAL_STEP_TC);
        }

    } // End of local global loop

    { // 3. Evaluate fullspace positions and subspace velocities
        PROFILE_STEP("FULLSPACE");
        if (_debug) spdlog::info("> HRPDSimulator::Step() - FULLSPACE");
        _sub.InterpolateSubspaceToFullspafceForPos(m_mesh->m_positions, m_positionsSubspace, false);
        // Util::storeData(m_mesh->m_positions, "debug/STEP/fullPos");
        // Not need to update full space velocities, just the subapce velocities is enough.
        m_velocitiesSubspace = (m_positionsSubspace - oldPosSub) / m_dt;
    }

    m_frameCount++;
}

void HRPDSimulator::Reset()
{
    { // reset pos, vel, ...
        m_mesh->ResetPositions();
        m_positionsSubspace = m_restPositionsSubspace;
        m_velocitiesSubspace.setZero();
        m_collisionCorrection = false;
        m_subspaceBuilder.m_velocitiesUsedVsRepulsion.setZero();
    }

    { // reset showdata
        m_innerForceInterpol.setZero();
        m_pdE.setZero();
        m_stvkE.setZero();
        m_innerForceFullSTVK.setZero();
        m_innerForceFullPD.setZero();
    }
}

void HRPDSimulator::ComputeWeightedExtForces()
{
    PROFILE_PREC("ComputeWeightedExtForces");
    // When change f_{ext}, need to call this.
    spdlog::info(">>> HRPD-SubspaceBuilder::ComputeWeightedExtForces");
    int _N = m_mesh->m_positions.rows();
    auto& sub = m_subspaceBuilder;
    PDPositions _fExtWeighted(_N, 3);
    PDPositions _fGravityWeighted(_N, 3);
    PD_PARALLEL_FOR
    for (int v = 0; v < _N; v++) {
        for (int d = 0; d < 3; d++) {
            _fExtWeighted(v, d) = m_fExt(v, d) * (1. / m_mesh->m_vertexMasses(v));
            _fGravityWeighted(v, d) = m_fGravity(v, d) * (1. / m_mesh->m_vertexMasses(v));
        }
    }
    sub.ProjectFullspaceToSubspaceForPos(m_fExtWeightedSubspace, _fExtWeighted);
    sub.ProjectFullspaceToSubspaceForPos(m_fGravityWeightedSubspace, _fGravityWeighted);
}

// -------------------------------------- Collision Handle --------------------------------------

void HRPDSimulator::handleGripAndCollisionsUsedVs(PDPositions& s, bool update)
{
    auto& sub = m_subspaceBuilder;
    m_collisionCorrection = false;
    bool gripCorrection = false;
    auto& pos = sub.m_positionsUsedVs;
    auto& velRepulsion = sub.m_velocitiesUsedVsRepulsion;
    PD_PARALLEL_FOR
    for (int v = 0; v < sub.m_positionsUsedVs.rows(); v++) {
        velRepulsion.row(v) = pos.row(v);
        // -- Floor
        if (g_InteractState.isFloorActive) {
            if (pos(v, 1) < g_InteractState.floorHeight) {
                m_collisionCorrection = true;
                pos(v, 1) = g_InteractState.floorHeight;
            }
        }
        // -- Collision
        // for (auto& obj : m_operationObjects) {
        //     auto colObj = std::dynamic_pointer_cast<Entity::CollisionObject>(obj);
        //     if (colObj != nullptr) {
        //         PD3dVector posV = pos.row(v);
        //         if (colObj->resolveCollision(posV)) {
        //             m_collisionCorrection = true;
        //             pos.row(v) = posV.transpose();
        //         }
        //     }
        // }
        // -- Grip
        // for (auto& obj : m_operationObjects) {
        //     auto gripObj = std::dynamic_pointer_cast<Entity::GripObject>(obj);
        //     if (gripObj != nullptr) {
        //         PD3dVector posV = pos.row(v);
        //         PD3dVector initPosV = m_restPositions.row(m_usedVertices[v]);
        //         if (gripObj->resolveGrip(posV, initPosV)) {
        //             gripCorrection = true;
        //             pos.row(v) = posV.transpose();
        //         }
        //     }
        // }
        velRepulsion.row(v) = 1.0 * (pos.row(v) - velRepulsion.row(v));
    }

    // User interaction - Drag
    if (g_InteractState.draggingState.isDragging) {
        if (g_InteractState.draggingState.vertex != -1) {
            if (g_InteractState.draggingState.vertexUsed == -1) {
                for (int i = 0; i < sub.m_positionsUsedVs.rows(); i++) {
                    if (g_InteractState.draggingState.vertexUsed == -1) {
                        g_InteractState.draggingState.vertexUsed = i;
                        continue;
                    }
                    auto x = GetPositions().row(g_InteractState.draggingState.vertex);
                    auto prev = sub.m_positionsUsedVs.row(g_InteractState.draggingState.vertexUsed);
                    auto curr = sub.m_positionsUsedVs.row(i);
                    if ((x - prev).norm() > (x - curr).norm()) {
                        g_InteractState.draggingState.vertexUsed = i;
                    }
                }
            }

            int vs = g_InteractState.draggingState.vertexUsed;
            // double dt2 = g_InteractState.timeStep * g_InteractState.timeStep;
            // m_positionsUsedVs.row(v) += (1.0 / dt2) * g_InteractState.draggingState.force.cast<PDScalar>();
            sub.m_positionsUsedVs.row(vs) += g_InteractState.draggingState.force.cast<PDScalar>();
        }
    }

    // If collisions were resolved, update s in the subspace via interpolation
    // of the corrected vertices
    if (m_collisionCorrection || gripCorrection || g_InteractState.draggingState.isDragging) {

        sub.ProjectUsedFullspaceToSubspaceForPos(s, sub.m_positionsUsedVs);
        if (update) m_positionsSubspace = s;
        Eigen::RowVector3d _old = sub.m_positionsUsedVs.row(g_InteractState.draggingState.vertexUsed);
        sub.InterpolateSubspaceToFullspafceForPos(sub.m_positionsUsedVs, m_positionsSubspace, true);
        Eigen::RowVector3d _new = sub.m_positionsUsedVs.row(g_InteractState.draggingState.vertexUsed);
        // spdlog::critical(">>> Handle - DRAG Vertex DIFF Module = {}", (_new - _old).norm());
    }
}

// -------------------------------------------------------------------------------------------------

void HRPDSimulator::updateLHSMatrix()
{
    PDMatrix lhsMatrix = m_subspaceLHS_mom * m_dt2inv + (g_InteractState.hrpdParams.wi / m_mesh->m_stiffness_weight) * m_subspaceLHS_inner;

    // m_subspaceSystemSolverDense.compute(lhsMatrix);
    // if (m_subspaceSystemSolverDense.info() != Eigen::Success) {
    //     spdlog::warn("Warning: Factorization of LHS matrix for global system was not successful!");
    // }

    PDSparseMatrix lhsMatrixSampledSparse = lhsMatrix.sparseView(0, PD_SPARSITY_CUTOFF);
    m_subspaceSystemSolverSparse.compute(lhsMatrixSampledSparse);
    if (m_subspaceSystemSolverSparse.info() != Eigen::Success) {
        spdlog::warn("Warning: Factorization of the sparse LHS matrix for the global step was not successful!");
        PDSparseMatrix eps(lhsMatrixSampledSparse.rows(), lhsMatrixSampledSparse.rows());
        eps.setIdentity();
        eps *= 1e-12;
        while (m_subspaceSystemSolverSparse.info() != Eigen::Success && eps.coeff(0, 0) < 1e-10) {
            spdlog::warn("Adding small diagonal entries ({})...", eps.coeff(0, 0));
            lhsMatrixSampledSparse += eps;
            eps *= 2;
            m_subspaceSystemSolverSparse.compute(lhsMatrixSampledSparse);
        }
    }
}

void HRPDSimulator::UpdateStiffnessWeight()
{
    // spdlog::info(">>> m_mesh->m_weight = {}", m_mesh->m_weight);
    // spdlog::info(">>> g_InteractState.hrpdParams.wi * m_normalizationMass = {}", g_InteractState.hrpdParams.wi * m_normalizationMass);
    // m_mesh->m_stiffness_weight = g_InteractState.hrpdParams.wi;
    updateLHSMatrix();
}

void HRPDSimulator::UpdateTimeStep()
{
    m_dt = g_InteractState.timeStep;
    m_dt2 = m_dt * m_dt;
    m_dt2inv = 1.0 / m_dt2;

    updateLHSMatrix();
}

void HRPDSimulator::SetGravity(PDScalar gravityConstant)
{
    spdlog::info(">>> HRPDSimulator::SetGravity() ");
    m_fGravity.setZero(m_mesh->m_positions.rows(), 3);
    PD_PARALLEL_FOR
    for (int v = 0; v < m_fGravity.rows(); v++) {
        m_fGravity(v, 1) = -gravityConstant * m_mesh->m_vertexMasses(v);
    }
    ComputeWeightedExtForces();
}

}