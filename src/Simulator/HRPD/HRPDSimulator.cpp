
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
#include "spdlog/spdlog.h"

#ifdef PD_USE_CUDA
#include "CUDA/CUDAMatrixOP.hpp"
#endif

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
#ifdef PD_USE_CUDA
        PDSparseMatrix mid = _sub.m_ST * _sub.m_weightMatrixForProj * _sub.m_ST.transpose();
        CUDAMatrixUTMU(_sub.m_U, mid, m_subspaceLHS_inner);
#else
        m_subspaceLHS_inner += (_sub.m_UT * (_sub.m_ST * _sub.m_weightMatrixForProj * _sub.m_ST.transpose()) * _sub.m_U);
#endif
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
    ms_prevPositionsSub = m_positionsSubspace;
    bool _debug = false;

    auto& _sub = m_subspaceBuilder;
    // 1. Get inertia s
    {
        PROFILE_STEP("GET_INERTIA_s");
        if (_debug) spdlog::info("> HRPDSimulator::Step() - GET_INERTIA_s");
        if (m_collisionCorrection) {
            if (g_InteractState.isUseFullspaceVerticesForCollisionHandling) {
                auto& vel = m_mesh->m_velocities;
                auto& velRepulsion = _sub.m_velocitiesRepulsion;
                // Remove tangential movement and add repulsion movement from collided vertices
                PD_PARALLEL_FOR
                for (int v = 0; v < vel.rows(); v++) {
                    if (velRepulsion.row(v).norm() > 1e-12) {
                        PD3dVector curV = vel.row(v);
                        PD3dVector corrV = velRepulsion.row(v);
                        PDVector tangentialV = curV - curV.dot(corrV) * corrV;
                        tangentialV *= (1.0 - m_frictionCoeff);
                        tangentialV += corrV * m_repulsionCoeff;
                        vel.row(v) = tangentialV;
                    }
                }
                // Project back to subspace
                _sub.ProjectFullspaceToSubspaceForPos(m_velocitiesSubspace, vel);
            }
            else {
                // Get actual velocities for used vertices
                _sub.InterpolateSubspaceToFullspaceForPos(_sub.m_velocitiesUsedVs, m_velocitiesSubspace, true);
                // Remove tangential movement and add repulsion movement from collided vertices
                PD_PARALLEL_FOR
                for (int v = 0; v < _sub.m_velocitiesUsedVs.rows(); v++) {
                    if (_sub.m_velocitiesUsedVsRepulsion.row(v).norm() > 1e-12) {
                        PD3dVector curV = _sub.m_velocitiesUsedVs.row(v);
                        PD3dVector corrV = _sub.m_velocitiesUsedVsRepulsion.row(v);
                        PD3dVector corrVDir = corrV.normalized();
                        PDVector tangentialV = curV - curV.dot(corrVDir) * corrVDir;
                        tangentialV *= (1.0 - m_frictionCoeff);
                        tangentialV += corrV * m_repulsionCoeff;
                        _sub.m_velocitiesUsedVs.row(v) = tangentialV;
                    }
                }
                // Project back to subspace
                _sub.ProjectUsedFullspaceToSubspaceForPos(m_velocitiesSubspace, _sub.m_velocitiesUsedVs);
            }
            m_collisionCorrection = false;
        }
        ms_s = m_positionsSubspace + m_dt * m_velocitiesSubspace + m_dt2 * (m_fExtWeightedSubspace + m_fGravityWeightedSubspace);
        if (m_rayleighDampingAlpha > 0) {
            ms_s -= m_dt * m_rayleighDampingAlpha * m_velocitiesSubspace;
        }
        m_positionsSubspace = ms_s;
        // Get actual positions for used vertices
        if (g_InteractState.isUseFullspaceVerticesForCollisionHandling) {
            _sub.InterpolateSubspaceToFullspaceForPos(m_mesh->m_positions, m_positionsSubspace, false);
        }
        else {
            _sub.InterpolateSubspaceToFullspaceForPos(_sub.m_positionsUsedVs, m_positionsSubspace, true);
        }
        // Correct vertex positions of gripped and collided vertices
        handleGripAndCollisionsUsedVs(ms_s, true);
    }

    { // rhs0 - (UT @ M @ U) /h^2 @ s
        PROFILE_STEP("GET_RHS0");
        rhs0 = m_subspaceRHSSparse_mom * m_dt2inv * ms_s;
    }
    // 2. Local / global loop
    for (int i = 0; i < g_InteractState.numIterations; i++) {
        { // 2.1. Local step: Constraint projections
            PROFILE_STEP("LOCAL_ITER");
            if (_debug) spdlog::info("> HRPDSimulator::Step() - LOCAL_ITER [{}]", std::to_string(i));
            // rhs2 - (UT @ M/h^2 @ U @ s) + (UT @ ST @ WI @ V @ subP)
            // TICK(LOCAL_STEP_TC);
            rhs2 = rhs0;
            // for (auto& sub : m_subspaceBuilders) {
            {
                auto& sub = m_subspaceBuilder;
                VTJTPused.resize(sub.m_V.cols(), 3);
                VPsub.resize(sub.m_V.cols(), 3);
                sub.InterpolateSubspaceToFullspaceForPos(_sub.m_positionsUsedVs, m_positionsSubspace, true);
                // Collect auxiliary variables for sampled constraints
                sub.GetUsedP();
                // Set up rhs from auxiliaries and solve the system for all three columns
                PD_PARALLEL_FOR
                for (int d = 0; d < 3; d++) {
                    // VT * JT * P_{partial}
                    VTJTPused.col(d) = sub.ms_VTJT * sub.m_projectionsUsedVs.col(d);
                    // V * P_{sub}
                    VPsub.col(d) = sub.m_fittingSolver.solve(VTJTPused.col(d));

                    // Apply finalization matrix and add to current rhs
                    rhs2.col(d) += (g_InteractState.hrpdParams.wi / m_mesh->m_stiffness_weight) * sub.ms_UTSTWiV * VPsub.col(d);
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
        _sub.InterpolateSubspaceToFullspaceForPos(m_mesh->m_positions, m_positionsSubspace, false);
        // Util::storeData(m_mesh->m_positions, "debug/STEP/fullPos");
        // Not need to update full space velocities, just the subapce velocities is enough.
        m_velocitiesSubspace = (m_positionsSubspace - ms_prevPositionsSub) / m_dt;
    }

    { // Operation objects step
        PROFILE_STEP("OP_OBJECTS_STEP");
        if (_debug) spdlog::info("> HRPDSimulator::Step() - OP_OBJECTS_STEP");
        m_OpManager.Step(m_dt);
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

    {
        m_OpManager.Reset();
    }

    {
        m_innerForceInterpol.setZero();
        m_PD_E.setZero();
        m_STVK_E.setZero();
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
    PROFILE_STEP("COLLISION_HANDLE");
    auto& sub = m_subspaceBuilder;
    m_collisionCorrection = false;
    bool gripCorrection = false;
    auto& pos = sub.m_positionsUsedVs;
    auto& velRepulsion = sub.m_velocitiesUsedVsRepulsion;
    if (g_InteractState.isUseFullspaceVerticesForCollisionHandling) {
        pos = m_mesh->m_positions;
        velRepulsion = sub.m_velocitiesRepulsion;
    }
    PD_PARALLEL_FOR
    for (int v = 0; v < pos.rows(); v++) {
        velRepulsion.row(v) = pos.row(v);
        // -- Floor
        if (g_InteractState.isFloorActive) {
            if (pos(v, 1) < g_InteractState.floorHeight) {
                m_collisionCorrection = true;
                pos(v, 1) = g_InteractState.floorHeight;
            }
        }
        // -- Collision
        for (auto& obj : m_OpManager.m_operationObjects) {
            auto colObj = std::dynamic_pointer_cast<CollisionObject>(obj);
            if (colObj != nullptr) {
                PD3dVector posV = pos.row(v);
                if (colObj->ResolveCollision(posV)) {
                    m_collisionCorrection = true;
                    pos.row(v) = posV.transpose();
                }
            }
        }
        // -- Grip
        for (auto& obj : m_OpManager.m_operationObjects) {
            auto gripObj = std::dynamic_pointer_cast<GripObject>(obj);
            if (gripObj != nullptr) {
                PD3dVector posV = pos.row(v);
                PD3dVector prevPosV = m_mesh->m_positions.row(m_subspaceBuilder.m_usedVertices[v]);
                if (gripObj->ResolveGrip(posV)) {
                    gripCorrection = true;
                    pos.row(v) = prevPosV.transpose();
                }
            }
        }
        velRepulsion.row(v) = 1.0 * (pos.row(v) - velRepulsion.row(v));
    }

    // User interaction - Drag
    if (g_InteractState.draggingState.isDragging) {
        if (g_InteractState.draggingState.vertex != -1) {
            if (g_InteractState.isUseFullspaceVerticesForCollisionHandling) { // Update fullspace vertex
                m_mesh->m_positions.row(g_InteractState.draggingState.vertex) += g_InteractState.draggingState.force.cast<PDScalar>();
            }
            else { // Update subspace used vertex
                int vertexUsed = -1;
                {
                    for (int i = 0; i < sub.m_positionsUsedVs.rows(); i++) {
                        if (vertexUsed == -1) {
                            vertexUsed = i;
                            continue;
                        }
                        auto x = GetPositions().row(g_InteractState.draggingState.vertex);
                        auto prev = sub.m_positionsUsedVs.row(vertexUsed);
                        auto curr = sub.m_positionsUsedVs.row(i);
                        if ((x - prev).norm() > (x - curr).norm()) {
                            vertexUsed = i;
                        }
                    }
                }

                int vs = vertexUsed;
                // double dt2 = g_InteractState.timeStep * g_InteractState.timeStep;
                // m_positionsUsedVs.row(v) += (1.0 / dt2) * g_InteractState.draggingState.force.cast<PDScalar>();
                sub.m_positionsUsedVs.row(vs) += g_InteractState.draggingState.force.cast<PDScalar>();
            }
        }
    }

    // If collisions were resolved, update s in the subspace via interpolation
    // of the corrected vertices
    if (m_collisionCorrection || gripCorrection || g_InteractState.draggingState.isDragging) {

        // sub.ProjectUsedFullspaceToSubspaceForPos(s, sub.m_positionsUsedVs);
        // if (update) m_positionsSubspace = s;
        // Eigen::RowVector3d _old = sub.m_positionsUsedVs.row(g_InteractState.draggingState.vertexUsed);
        // sub.InterpolateSubspaceToFullspaceForPos(sub.m_positionsUsedVs, m_positionsSubspace, true);
        // Eigen::RowVector3d _new = sub.m_positionsUsedVs.row(g_InteractState.draggingState.vertexUsed);
        // spdlog::critical(">>> Handle - DRAG Vertex DIFF Module = {}", (_new - _old).norm());

        if (g_InteractState.isUseFullspaceVerticesForCollisionHandling) {
            sub.ProjectFullspaceToSubspaceForPos(s, m_mesh->m_positions);
            if (update) m_positionsSubspace = s;
            sub.InterpolateSubspaceToFullspaceForPos(sub.m_positionsUsedVs, m_positionsSubspace, true);
        }
        else {
            sub.ProjectUsedFullspaceToSubspaceForPos(s, sub.m_positionsUsedVs);
            if (update) m_positionsSubspace = s;
            sub.InterpolateSubspaceToFullspaceForPos(sub.m_positionsUsedVs, m_positionsSubspace, true);
        }
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

Eigen::MatrixXd HRPDSimulator::GetColorMapData()
{
    if (m_colorMapType == 0) { // DISPLACEMENT
        return (m_mesh->m_positions - m_mesh->m_restpose_positions).cast<double>().rowwise().norm();
    }
    else if (m_colorMapType == 1) { // INNER_FORCE_INTERPOL
        // todo
        auto& _sub = m_subspaceBuilder;
        // rhs2 - (UT @ M/h^2 @ U @ s) + (UT @ ST @ WI @ V @ subP)
        // rhs1 - UT @ ST @ WI @ V @ subP
        rhs1.setZero(_sub.m_U.cols(), 3);
        {
            auto& sub = m_subspaceBuilder;
            VTJTPused.resize(sub.m_V.cols(), 3);
            VPsub.resize(sub.m_V.cols(), 3);
            sub.InterpolateSubspaceToFullspaceForPos(_sub.m_positionsUsedVs, m_positionsSubspace, true);
            // Collect auxiliary variables for sampled constraints
            sub.GetUsedP();
            // Set up rhs from auxiliaries and solve the system for all three columns
            PD_PARALLEL_FOR
            for (int d = 0; d < 3; d++) {
                // VT * JT * P_{partial}
                VTJTPused.col(d) = sub.ms_VTJT * sub.m_projectionsUsedVs.col(d);
                // V * P_{sub}
                VPsub.col(d) = sub.m_fittingSolver.solve(VTJTPused.col(d));

                // Apply finalization matrix and add to current rhs
                rhs1.col(d) += (g_InteractState.hrpdParams.wi / m_mesh->m_stiffness_weight) * sub.ms_UTSTWiV * VPsub.col(d);
            }
        }
        // - ( UT * (ST * WI * S) * U  -  UT @ ST @ WI @ V @ subP)
        m_innerForceSubspace = -(m_subspaceLHS_inner * m_positionsSubspace - rhs1);
        m_innerForceInterpol.resize(_sub.m_U_sp.rows(), 3);
        PD_PARALLEL_FOR
        for (int d = 0; d < 3; d++) {
            m_innerForceInterpol.col(d) = _sub.m_U_sp * m_innerForceSubspace.col(d);
        }
        if (isUniformColorMap) {
            auto tmp = m_innerForceInterpol;
            PD_PARALLEL_FOR
            for (int v = 0; v < m_mesh->m_positions.rows(); v++) {
                m_innerForceInterpol.row(v).setZero();
                for (auto nv : m_mesh->m_adjVerts2rd[v]) {
                    m_innerForceInterpol.row(v) += tmp.row(nv);
                }
                m_innerForceInterpol.row(v) /= (m_mesh->m_adjVerts2rd[v].size() + 1);
            }
        }

        return m_innerForceInterpol;
    }
    else if (m_colorMapType == 2) { // PD_ENERGY
        m_PD_E.setZero(m_mesh->m_positions.rows(), 1);
        std::vector<PDScalar> tetsE(m_mesh->m_tets.rows());
        PD_PARALLEL_FOR
        for (int tInd = 0; tInd < m_mesh->m_tets.rows(); tInd++) {
            PDPositions pos(4, 3);
            for (int k = 0; k < 4; k++) {
                pos.row(k) = m_mesh->m_positions.row(m_mesh->m_tets(tInd, k));
            }
            tetsE[tInd] = m_mesh->GetPDEnergy(pos, tInd);
        }
        PD_PARALLEL_FOR
        for (int v = 0; v < m_mesh->m_positions.rows(); v++) {
            for (auto tp : m_mesh->m_tetsPerVertex[v]) {
                m_PD_E(v, 0) += tetsE[tp.first];
            }
            m_PD_E(v, 0) /= 4.0;
        }
        if (isUniformColorMap) {
            auto tmp = m_PD_E;
            PD_PARALLEL_FOR
            for (int v = 0; v < m_mesh->m_positions.rows(); v++) {
                m_PD_E.row(v).setZero();
                for (auto nv : m_mesh->m_adjVerts2rd[v]) {
                    m_PD_E.row(v) += tmp.row(nv);
                }
                m_PD_E.row(v) /= (m_mesh->m_adjVerts2rd[v].size() + 1);
            }
        }
        return m_PD_E.rowwise().norm();
    }
    else if (m_colorMapType == 3) { // STVK_ENERGY
        std::vector<PDScalar> tetsE(m_mesh->m_tets.rows());
        PD_PARALLEL_FOR
        for (int tInd = 0; tInd < m_mesh->m_tets.rows(); tInd++) {
            PDPositions pos(4, 3);
            for (int k = 0; k < 4; k++) {
                pos.row(k) = m_mesh->m_positions.row(m_mesh->m_tets(tInd, k));
            }
            tetsE[tInd] = m_mesh->GetStvkEnergy(pos, tInd);
        }
        m_STVK_E.resize(m_mesh->m_positions.rows(), 1);
        PD_PARALLEL_FOR
        for (int v = 0; v < m_mesh->m_positions.rows(); v++) {
            m_STVK_E(v, 0) = 0;
            for (auto tp : m_mesh->m_tetsPerVertex[v]) {
                m_STVK_E(v, 0) += tetsE[tp.first];
            }
            m_STVK_E(v, 0) /= 4.0;
        }
        if (isUniformColorMap) {
            auto tmp = m_STVK_E;
            PD_PARALLEL_FOR
            for (int v = 0; v < m_mesh->m_positions.rows(); v++) {
                m_STVK_E.row(v).setZero();
                for (auto nv : m_mesh->m_adjVerts2rd[v]) {
                    m_STVK_E.row(v) += tmp.row(nv);
                }
                m_STVK_E.row(v) /= (m_mesh->m_adjVerts2rd[v].size() + 1);
            }
        }
        return m_STVK_E.rowwise().norm();
    }
    else {
        // todo
        return Eigen::MatrixXd::Zero(m_mesh->m_positions.rows(), 1);
    }
}

}