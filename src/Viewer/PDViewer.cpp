#include "Viewer/PDViewer.hpp"

#include "Simulator/PDTypeDef.hpp"
#include "Simulator/HRPD/HRPDSimulator.hpp"
#include "Simulator/QNPD/QNPDSimulator.hpp"
#include "Simulator/QNPD/QNPDTetMesh.hpp"
#include "UI/InteractState.hpp"
#include "Util/Profiler.hpp"
#include "Util/Timer.hpp"

#include "igl/jet.h"
#include "igl/unproject_onto_mesh.h"
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>
#include "igl/png/writePNG.h"

#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"

#include <spdlog/spdlog.h>
#include <chrono>
#include <cstddef>

extern UI::InteractState g_InteractState;
extern igl::opengl::glfw::Viewer g_Viewer;
extern igl::opengl::glfw::imgui::ImGuizmoWidget g_Gizmo;

namespace PD {

PDViewer::PDViewer()
{
    if (g_InteractState.simulatorType == "HRPD") {
        spdlog::info("### USE HRPD SIMULATOR");
        auto tetMesh = std::make_shared<HRPDTetMesh>(g_InteractState.meshURL);
        m_sim = std::make_shared<HRPDSimulator>(tetMesh);
    }
    if (g_InteractState.simulatorType == "QNPD") {
        spdlog::info("### USE QNPD SIMULATOR [TODO]");
        auto tetMesh = std::make_shared<QNPDTetMesh>(g_InteractState.meshURL);
        m_sim = std::make_shared<QNPDSimulator>(tetMesh);
    }
    if (g_InteractState.simulatorType == "Newton") {
        spdlog::info("### USE Newton SIMULATOR [TODO]");
        auto tetMesh = std::make_shared<CL::TetMesh>(g_InteractState.meshURL);
        m_sim = std::make_shared<CL::NewtonSimulator>(tetMesh);
    }
    if (!m_sim) {
        spdlog::error("### SIMULATOR CREATE ERROR");
    }
}

void PDViewer::init(igl::opengl::glfw::Viewer* _viewer)
{
    viewer = _viewer;

    m_imguiContext = std::make_unique<ImGuiContext>();
    m_imguiContext->init(viewer->window);

    m_frameBuffer = std::make_unique<OpenGLFrameBuffer>();
    {
        // glfwGetWindowSize(viewer->window, &m_sceneWindowSize.first, &m_sceneWindowSize.second);
        // spdlog::info(">>> Window m_sceneWindowSize = ({}, {})", m_sceneWindowSize.first, m_sceneWindowSize.second);
        // viewer->core().viewport = { 0, 0, m_sceneWindowSize.first * 1.0f, m_sceneWindowSize.second * 1.0f };
        // spdlog::info(">>> viewer->core().viewport = ({}, {}, {}, {})", viewer->core().viewport(0), viewer->core().viewport(1), viewer->core().viewport(2), viewer->core().viewport(3));
        // m_frameBuffer->create_buffers(m_sceneWindowSize.first, m_sceneWindowSize.second);
    }
}

bool PDViewer::pre_draw()
{
    PROFILE("FRAME");
    m_frameTimer.startStopWatch();
    bool _DB = false;
    if (_DB) spdlog::info(">>> PDViewer::Frame");

    {
        PROFILE("PRE_DRAW");
        m_predrawTimer.startStopWatch();
        { // imgui context
            m_imguiContext->pre_render();
            m_frameBuffer->bind();
        }
        { // simulator step
            PROFILE("SIMULATOR_STEP");
            if (!m_sim.get()) {
                spdlog::error("ERROR: PDViewer::pre_draw - Not find simulator instance!");
                return false;
            }
            if (viewer->core().is_animating || (g_InteractState.isSingleStep)) {
                g_InteractState.isSingleStep = false;
                m_sim->Step();
            }
        }

        { // update mesh data
            PROFILE("IGL_SetMesh");
            m_sim->IGL_SetMesh(viewer);
        }
        {
            viewer->data(m_interactMeshID).clear();
            if (g_InteractState.draggingState.isDragging && (viewer->core().is_animating || g_InteractState.isSingleStep)) {
                auto vert = g_InteractState.draggingState.vertex;
                // spdlog::info("Pre draw - g_InteractState.draggingState points: {0} {1} {2}", pos.row(vert).x(), pos.row(vert).y(), pos.row(vert).z());
                viewer->data(m_interactMeshID).point_size = 20.f;
                auto C = Eigen::RowVector3d(0, 1, 0);
                viewer->data(m_interactMeshID).set_points(m_sim->GetPositions().cast<double>().row(vert), C);
            }
        }
        m_predrawTimer.stopStopWatch();
        if (_DB) spdlog::info(">>> PDViewer::Frame - After pre draw");
    }

    { // viewer core draw mesh
        PROFILE("DRAW_MESH");
        m_drawTimer.startStopWatch();
        for (auto& core : viewer->core_list) {
            for (auto& mesh : viewer->data_list) {
                if (mesh.is_visible & core.id) {
                    core.draw(mesh);
                }
            }
        }
        g_Gizmo.draw();
        m_drawTimer.stopStopWatch();
        if (_DB) spdlog::info(">>> PDViewer::Frame - After draw");
    }

    {
        PROFILE("POST_DRAW");
        m_postdrawTimer.startStopWatch();
        m_frameBuffer->unbind();

        ImVec2 viewportPanelSize;
        { // Scene window
            ImGui::Begin("Scene");
            viewportPanelSize = ImGui::GetContentRegionAvail();
            m_sceneWindowPos = ImGui::GetWindowPos();
            m_sceneCursorPos = ImGui::GetCursorPos();
            GLuint textureID = m_frameBuffer->get_texture();
            if (textureID) {
                ImGui::Image((void*)(intptr_t)(textureID), ImVec2{ viewportPanelSize.x, viewportPanelSize.y }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
            }
            isSceneInteractionActive = ImGui::IsItemHovered();
            ImGui::End();
            if (_DB) spdlog::info(">>> PDViewer::Frame - After ImGui::Image()");
        }
        // Render window
        {
            SimulatorInfoWindow();
            OperationWindow();
            ProfilerWindow();
        }
        {
            m_imguiContext->post_render();
            if (_DB) spdlog::info(">>> PDViewer::Frame - After imgui context post_render()");
            if (!m_frameBuffer->get_texture() || (viewportPanelSize.x != m_sceneWindowSize.first || viewportPanelSize.y != m_sceneWindowSize.second)) {
                m_sceneWindowSize = { viewportPanelSize.x, viewportPanelSize.y };
                viewer->core().viewport = { 0, 0, m_sceneWindowSize.first * 1.0f, m_sceneWindowSize.second * 1.0f };
                spdlog::info(">>> viewer->core().viewport = ({}, {}, {}, {})", viewer->core().viewport(0), viewer->core().viewport(1), viewer->core().viewport(2), viewer->core().viewport(3));
                spdlog::info(">>> Window m_sceneWindowSize = ({}, {})", m_sceneWindowSize.first, m_sceneWindowSize.second);
                m_frameBuffer->create_buffers(m_sceneWindowSize.first, m_sceneWindowSize.second);
            }
        }
        m_postdrawTimer.stopStopWatch();
        if (_DB) spdlog::info(">>> PDViewer::Frame - After post draw");
    }

    { // Camera test
        std::cout << "=================================================\n";
        std::cout << "Camera Info:\n";
        std::cout << "  camera_center: " << viewer->core().camera_center.transpose() << "\n";
        std::cout << "  camera_translation: " << viewer->core().camera_translation.transpose() << "\n";
        std::cout << "  camera_eye: " << viewer->core().camera_eye.transpose() << "\n";
        std::cout << "  camera_up: " << viewer->core().camera_up.transpose() << "\n";
        std::cout << "  camera_view_angle: " << viewer->core().camera_view_angle << "\n";
        std::cout << "  camera_base_zoom: " << viewer->core().camera_base_zoom << "\n";
        std::cout << "  camera_zoom: " << viewer->core().camera_zoom << "\n";
        std::cout << "  camera_dnear: " << viewer->core().camera_dnear << "\n";
        std::cout << "  camera_dfar: " << viewer->core().camera_dfar << "\n";
        std::cout << "  trackball_angle: " << viewer->core().trackball_angle << "\n";
        std::cout << "=================================================\n";
    }

    m_frameTimer.stopStopWatch();
    return true;
}

void PDViewer::Setup()
{
    // -- After g_Viewer.launch_init() - assert(viewer!=null);
    // Config floor mesh for draw
    spdlog::info("> PDViewer::setup - Before");
    {
        if (m_floorMeshID == -1) {
            spdlog::info("> PDViewer::setup - m_floorMeshID = {}", m_floorMeshID);
            m_floorMeshID = viewer->append_mesh(g_InteractState.isFloorActive);
            spdlog::info("> PDViewer::setup - m_floorMeshID = {}", m_floorMeshID);
        }
        double gridSize = m_floorGridSize;
        double step = gridSize / 25;
        Eigen::RowVector3d C;
        for (double f = -gridSize; f <= gridSize; f += step) {
            double y = 0.;
            Eigen::RowVector3d p1 = { f, y, -gridSize };
            Eigen::RowVector3d p2 = { f, y, gridSize };
            Eigen::RowVector3d p3 = { -gridSize, y, f };
            Eigen::RowVector3d p4 = { gridSize, y, f };
            if (std::abs(f) < 1e-12)
                C = { 0.0, 0.0, 0.0 };
            else
                C = { 0.8, 0.8, 0.8 };
            viewer->data(m_floorMeshID).add_edges(p1, p2, C);
            viewer->data(m_floorMeshID).add_edges(p3, p4, C);
        }
    }
    {
        if (m_interactMeshID == -1) {
            m_interactMeshID = viewer->append_mesh(true);
            spdlog::info("> PDViewer::setup - m_interactMeshID = {}", m_interactMeshID);
        }
    }

    spdlog::info("> PDViewer::setup - After");
}

void PDViewer::Reset()
{
    viewer->core().is_animating = false;

    m_sim->Reset();
}

// ====================================================================================================
// ---------------- User Interaction ----------------
// ====================================================================================================

bool PDViewer::mouse_down(int button, int modifier)
{
    if (g_Gizmo.visible && ImGuizmo::IsOver()) return true;
    if (!isSceneInteractionActive) return true;
    if (modifier != GLFW_MOD_CONTROL) return false;
    float x = viewer->current_mouse_x - (m_sceneWindowPos.x - m_imguiContext->root_window_pos.x + m_sceneCursorPos.x);
    float y = viewer->core().viewport(3) - (viewer->current_mouse_y - (m_sceneWindowPos.y - m_imguiContext->root_window_pos.y + m_sceneCursorPos.y));
    g_InteractState.draggingState.prevMouse = Eigen::RowVector3d(x, y, 0.0);

    int fid;
    Eigen::Vector3f bc;
    bool hit = igl::unproject_onto_mesh(
        Eigen::Vector2f(x, y), viewer->core().view, viewer->core().proj,
        viewer->core().viewport, m_sim->GetPositions(), m_sim->GetTriangles(),
        fid, bc);
    if (hit) {
        long c;
        bc.maxCoeff(&c);
        g_InteractState.draggingState.vertex = m_sim->GetTriangles()(fid, c);
        g_InteractState.draggingState.force = Eigen::RowVector3d(0.0, 0.0, 0.0);
        g_InteractState.draggingState.isDragging = true;
        return true;
    }

    return false;
}

bool PDViewer::mouse_move(int button, int modifier)
{
    if (g_Gizmo.visible && ImGuizmo::IsOver()) return true;
    if (!isSceneInteractionActive) return true;
    if (!g_InteractState.draggingState.isDragging) return false;
    auto curr_mouse = Eigen::RowVector3d(
        viewer->current_mouse_x - (m_sceneWindowPos.x - m_imguiContext->root_window_pos.x + m_sceneCursorPos.x),
        viewer->core().viewport(3) - (viewer->current_mouse_y - (m_sceneWindowPos.y - m_imguiContext->root_window_pos.y + m_sceneCursorPos.y)),
        g_InteractState.draggingState.prevMouse(2));
    Eigen::RowVector3d prev_pos, curr_pos;

    igl::unproject(g_InteractState.draggingState.prevMouse, viewer->core().view,
        viewer->core().proj, viewer->core().viewport, prev_pos);
    igl::unproject(curr_mouse, viewer->core().view, viewer->core().proj,
        viewer->core().viewport, curr_pos);

    auto direction = (curr_pos - prev_pos).normalized();
    // spdlog::info("Direction: {} {} {}", direction(0), direction(1), direction(2));
    g_InteractState.draggingState.force = direction * g_InteractState.draggingState.forceStrength;
    g_InteractState.draggingState.prevMouse = curr_mouse;

    return false;
}

bool PDViewer::mouse_up(int button, int modifier)
{
    if (!isSceneInteractionActive) return true;
    g_InteractState.draggingState.isDragging = false;
    g_InteractState.draggingState.vertex = -1;
    return false;
}

// ====================================================================================================
// ---------------- Windows ----------------
// ====================================================================================================
void PDViewer::SimulatorInfoWindow()
{
    ImGui::Begin("SimulatorInfo");
    float const w = ImGui::GetContentRegionAvail().x;
    float const p = 0.0f;
    ImVec2 buttonSize = { (w - p) / 4.0f, 0 };

    { // Simulator info
        ImGui::Text("Simulator Type: %s", g_InteractState.simulatorType.c_str());
        ImGui::InputDouble("Simulator TimeStep", &g_InteractState.timeStep, 0.001f, 0.01f, "%.6f");
        if (ImGui::Button("Update TimeStep", { (w - p) * 0.5f, 0 })) {
            viewer->core().is_animating = false;
            m_sim->UpdateTimeStep();
        }
        ImGui::InputInt("Simulator Iterations", &g_InteractState.numIterations);
        if (g_InteractState.simulatorType == "HRPD") {
            auto hrpd_simulator = dynamic_cast<HRPDSimulator*>(m_sim.get());
            if (ImGui::CollapsingHeader("HRPD Params", ImGuiTreeNodeFlags_DefaultOpen)) {
                auto& params = g_InteractState.hrpdParams;
                // ImGui::InputDouble("Gravity", &params.gravityConstant, 0.001f, 0.01f, "%.6f");
                // ImGui::InputInt("numberSamplesForVertexPosSubspace", &params.numberSamplesForVertexPosSubspace);
                // ImGui::InputDouble("radiusMultiplierForPosSubspace", &params.radiusMultiplierForPosSubspace);
                // ImGui::InputInt("numberSamplesForVertexProjSubspace", &params.numberSamplesForVertexProjSubspace);
                // ImGui::InputDouble("radiusMultiplierForProjSubspace", &params.radiusMultiplierForProjSubspace);
                // ImGui::InputInt("numberSampledConstraints", &params.numberSampledConstraints);

                ImGui::BulletText("%s", std::string("numberSamplesForVertexPosSubspace : " + std::to_string(params.numberSamplesForVertexPosSubspace)).c_str());
                ImGui::BulletText("%s", std::string("radiusMultiplierForPosSubspace : " + std::to_string(params.radiusMultiplierForPosSubspace)).c_str());
                ImGui::BulletText("%s", std::string("numberSamplesForVertexProjSubspace : " + std::to_string(params.numberSamplesForVertexProjSubspace)).c_str());
                ImGui::BulletText("%s", std::string("radiusMultiplierForProjSubspace : " + std::to_string(params.radiusMultiplierForProjSubspace)).c_str());
                ImGui::BulletText("%s", std::string("numberSampledConstraints : " + std::to_string(params.numberSampledConstraints)).c_str());

                bool _isGravityActive = g_InteractState.isGravityActive;
                ImGui::Checkbox("Gravity Active", &g_InteractState.isGravityActive);
                if (_isGravityActive != g_InteractState.isGravityActive) {
                    Reset();
                    hrpd_simulator->SetGravity(g_InteractState.isGravityActive == true ? params.gravityConstant : 0.0);
                }
                if (g_InteractState.isGravityActive) {
                    ImGui::BulletText("%s", std::string("Gravity constant : " + std::to_string(params.gravityConstant)).c_str());
                }

                if (ImGui::CollapsingHeader("StrainLimiting", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::InputFloat("WI", &params.wi, 0.0f, 0.0f, "%.8f");
                    if (ImGui::Button("Update Stiffness WI", { (w - p) * 0.7f, 0 })) {
                        if (hrpd_simulator) {
                            viewer->core().is_animating = false;
                            hrpd_simulator->UpdateStiffnessWeight();
                        }
                    }
                    ImGui::InputFloat("sigmaMin", &params.sigmaMin);
                    ImGui::InputFloat("sigmaMax", &params.sigmaMax);
                }
            }
        }
        if (g_InteractState.simulatorType == "QNPD") {
            if (ImGui::CollapsingHeader("QNPD Params", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::InputDouble("Gravity", &g_InteractState.qnpdParams.gravityConstant, 0.001f, 0.01f, "%.6f");
                auto& params = g_InteractState.qnpdParams;
                ImGui::Text("Material Type: %s", params.materialType.c_str());
                ImGui::Text("Solver Type: %s", params.solverType.c_str());
                ImGui::Text("Integration Method: %s", params.integration_method.c_str());
                ImGui::Text("Optimization Method: %s", params.optimization_method.c_str());
                ImGui::InputDouble("stiffness_attachment##QNPDParams", &params.stiffness_attachment);
                ImGui::InputDouble("stiffness_stretch##QNPDParams", &params.stiffness_stretch);
                ImGui::InputDouble("stiffness_bending##QNPDParams", &params.stiffness_bending);
                ImGui::InputDouble("stiffness_kappa##QNPDParams", &params.stiffness_kappa);
                ImGui::InputDouble("stiffness_laplacian##QNPDParams", &params.stiffness_laplacian);
            }
        }
        if (ImGui::Button("Apply Params (Need ReCompute)", { (w - p) * 0.7f, 0 })) {
            viewer->core().is_animating = false;
            ImGui::Text("Recomputing...");
            m_sim->LoadParamsAndApply();
        }
    }

    { // Colormap for HRPD
        auto hrpd = dynamic_cast<HRPDSimulator*>(m_sim.get());
        if (hrpd) {
            if (ImGui::CollapsingHeader("ColorMap", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (int i = 0; i < hrpd->ColorMapTypeStrs.size(); i++) {
                    auto& str = hrpd->ColorMapTypeStrs[i];
                    if (hrpd->ColorMapTypeActive[str]) {
                        ImGui::RadioButton(str.c_str(), &hrpd->m_colorMapType, i);
                    }
                }
            }
        }
    }

    ImGui::End();
}

void PDViewer::OperationWindow()
{
    ImGui::Begin("Operation");
    float const w = ImGui::GetContentRegionAvail().x;
    // float const p = ImGui::GetStyle().FramePadding.x;
    float const p = 0.0f;
    ImVec2 buttonSize = { (w - p) / 4.0f, 0 };

    { // Start and Stop Simulate
        if (!viewer->core().is_animating) {
            if (ImGui::Button("Start Simulate", { (w - p), 0 })) {
                viewer->core().is_animating = true;
            }
        }
        else {
            if (ImGui::Button("Stop Simulate", { (w - p), 0 })) {
                viewer->core().is_animating = false;
            }
        }
    }

    { // Print FPS
        if (viewer->core().is_animating) {
            ImGui::Text("%s", std::string("FPS info:").c_str());
            int lastMs_total = m_frameTimer.lastMeasurement() > 0 ? m_frameTimer.lastMeasurement() / 1000 : 0;
            ImGui::BulletText("%s", std::string("Last ms for frame : " + std::to_string(lastMs_total) + " ms").c_str());
            int lastMs_predraw = m_predrawTimer.lastMeasurement() > 0 ? m_predrawTimer.lastMeasurement() / 1000 : 0;
            ImGui::BulletText("%s", std::string("Last ms for predraw : " + std::to_string(lastMs_predraw) + " ms").c_str());
            int lastMs_draw = m_drawTimer.lastMeasurement() > 0 ? m_drawTimer.lastMeasurement() / 1000 : 0;
            ImGui::BulletText("%s", std::string("Last ms for draw : " + std::to_string(lastMs_draw) + " ms").c_str());
            int lastMs_postdraw = m_postdrawTimer.lastMeasurement() > 0 ? m_postdrawTimer.lastMeasurement() / 1000 : 0;
            ImGui::BulletText("%s", std::string("Last ms for postdraw : " + std::to_string(lastMs_postdraw) + " ms").c_str());
            double _FPS = 1.0 / (m_frameTimer.lastMeasurement() / 1e6);
            ImGui::BulletText("%s", std::string("FPS: " + std::to_string(int(_FPS))).c_str());
        }
    }

    { // Step and Reset
        if (ImGui::Button("Step", { (w - p) * 0.5f, 0 })) {
            if (!viewer->core().is_animating) {
                g_InteractState.isSingleStep = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset", { (w - p) * 0.5f, 0 })) {
            Reset();
        }
    }

    { // Align Camera
        if (ImGui::Button("Align Camera", { (w - p) * 0.5f, 0 })) {
            viewer->core().align_camera_center(m_sim->GetPositions(), m_sim->GetTriangles());
        }
    }

    { // Drag Force
        if (ImGui::CollapsingHeader("Dragging", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputFloat("Dragging force strength", &g_InteractState.draggingState.forceStrength, 1.f, 10.f, "%.6f");
        }
    }

    { // Operation objects
        auto HRPD_sim = dynamic_cast<HRPDSimulator*>(m_sim.get());
        if (ImGui::CollapsingHeader("Operation Objects", ImGuiTreeNodeFlags_DefaultOpen) && HRPD_sim) {
            // obj list
            auto& objs = HRPD_sim->m_OpManager.m_operationObjects;
            auto& objSelectInd = HRPD_sim->m_OpManager.m_selectObjectIndex;
            if (objs.size()) {
                ImGui::Separator();
                for (int i = 0; i < objs.size(); i++) {
                    char buf[32];
                    sprintf(buf, "#%.3i | %s", i, "Object");
                    // .data() and .c_str() cannot work.
                    // std::string curLabel = std::string("### Object ") + std::to_string(i);
                    if (ImGui::Selectable(buf, objSelectInd == i)) {
                        if (objSelectInd == i) {
                            objSelectInd = -1;
                        }
                        else {
                            objSelectInd = i;
                            g_Gizmo.T = HRPD_sim->m_OpManager.m_operationObjects[objSelectInd]->m_gizmoT.cast<float>();
                        }
                    }
                }
            }

            // add & del obj
            static int objTypeInd = 0;
            ImGui::Combo("Type", &objTypeInd, OperationObjectTypeStrs, IM_ARRAYSIZE(OperationObjectTypeStrs));
            if (ImGui::Button(" + ", buttonSize)) {
                // spdlog::info(">>> Add object - type = {}", OperationObjectTypeStrs[objTypeInd]);
                HRPD_sim->m_OpManager.AddOperationObject(OperationObjectTypeStrs[objTypeInd]);
            }
            if (HRPD_sim->m_OpManager.m_selectObjectIndex != -1) {
                ImGui::SameLine();
                if (ImGui::Button("Del", buttonSize)) {
                    // HRPD_sim->m_OpManager.del...
                    HRPD_sim->m_OpManager.DelOperationObject(&g_Viewer);
                }
            }
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen) && HRPD_sim) {
            if (HRPD_sim->m_OpManager.m_selectObjectIndex == -1) {
                ImGui::TextDisabled("No entity selected.");
            }
            else {
                // gizmo operation type
                if (ImGui::RadioButton("Translate", g_Gizmo.operation == ImGuizmo::TRANSLATE))
                    g_Gizmo.operation = ImGuizmo::TRANSLATE;
                // Currently not use ROTATE
                // ImGui::SameLine();
                // if (ImGui::RadioButton("Rotate", g_Gizmo.operation ==ImGuizmo::ROTATE))
                //     g_Gizmo.operation = ImGuizmo::ROTATE;
                ImGui::SameLine();
                if (ImGui::RadioButton("Scale", g_Gizmo.operation == ImGuizmo::SCALE))
                    g_Gizmo.operation = ImGuizmo::SCALE;

                // ################### Object info ###################
                PD::PD3dVector curLength;
                // Center
                auto& entity = HRPD_sim->m_OpManager.m_operationObjects[HRPD_sim->m_OpManager.m_selectObjectIndex];
                float _center[3];
                for (int i = 0; i < 3; i++) _center[i] = entity->m_center(i);
                ImGui::InputFloat3("Object Center", _center);
                for (int i = 0; i < 3; i++) entity->m_center(i) = _center[i];

                // For GRIP_CUBE
                if (std::dynamic_pointer_cast<GripCube>(entity)) {
                    auto gripCube = std::dynamic_pointer_cast<GripCube>(entity);
                    // Scale length
                    float _len[3];
                    for (int i = 0; i < 3; i++) _len[i] = gripCube->m_length(i);
                    ImGui::InputFloat3("Cube Length", _len);
                    for (int i = 0; i < 3; i++) curLength(i) = _len[i];
                    gripCube->SetLength(curLength);
                }
                // For COLLISION_CUBE
                if (std::dynamic_pointer_cast<CollisionCube>(entity)) {
                    auto collisionCube = std::dynamic_pointer_cast<CollisionCube>(entity);

                    ImGui::Separator();
                    // Scale length
                    float _len[3];
                    for (int i = 0; i < 3; i++)
                        _len[i] = collisionCube->m_length(i);
                    ImGui::InputFloat3("Cube Length", _len);
                    for (int i = 0; i < 3; i++) curLength(i) = _len[i];
                    collisionCube->SetLength(curLength);
                    // Move - Rotate & Translate
                    if (collisionCube->m_isRotate == true) {
                        int rtInds[2];
                        rtInds[0] = collisionCube->m_rotateEdgeInds.first,
                        rtInds[1] = collisionCube->m_rotateEdgeInds.second;
                        ImGui::InputInt2("Rotate Inds", rtInds);
                        collisionCube->initRotate({ rtInds[0], rtInds[1] }, collisionCube->m_moveVel);
                    }
                    else {
                        float curDir[3];
                        for (int k = 0; k < 3; k++)
                            curDir[k] = collisionCube->m_translateDir(k);
                        ImGui::InputFloat3("Move Dir", curDir);
                        for (int k = 0; k < 3; k++) {
                            collisionCube->m_translateDir(k) = curDir[k];
                            if (collisionCube->m_translateDir(k) != 0) {
                                collisionCube->m_isTranslate = true;
                            }
                        }
                    }
                    ImGui::InputDouble("Move Vel", &collisionCube->m_moveVel);
                    ImGui::InputInt("Move Count", &collisionCube->m_maxMove);
                }
                // For COLLISION_SPHERE
                if (std::dynamic_pointer_cast<CollisionSphere>(entity)) {
                    auto collisionSphere = std::dynamic_pointer_cast<CollisionSphere>(entity);

                    ImGui::Separator();
                    // Scale length
                    float _len[3];
                    for (int i = 0; i < 3; i++)
                        _len[i] = collisionSphere->m_length(i);
                    ImGui::InputFloat3("Cube Length", _len);
                    for (int i = 0; i < 3; i++) curLength(i) = _len[i];
                    collisionSphere->SetLength(curLength);

                    float curDir[3];
                    for (int k = 0; k < 3; k++)
                        curDir[k] = collisionSphere->m_translateDir(k);
                    ImGui::InputFloat3("Move Dir", curDir);
                    for (int k = 0; k < 3; k++)
                        collisionSphere->m_translateDir(k) = curDir[k];

                    ImGui::InputDouble("Move Vel", &collisionSphere->m_moveVel);
                    ImGui::InputInt("Move Count", &collisionSphere->m_maxMove);
                }

                ImGui::Separator();
            }
        }

        // ####################################################################################
        if (ImGui::CollapsingHeader("Operation Setup",
                ImGuiTreeNodeFlags_DefaultOpen)) {
            std::string curOperationInfo = "Current Operation Preset: ";
            static int curPresetIndex = 0;
            curOperationInfo += OperationPresets[curPresetIndex];

            ImGui::BulletText("%s", curOperationInfo.c_str());

            ImGui::Combo("Preset", &curPresetIndex, OperationPresets, IM_ARRAYSIZE(OperationPresets));

            if (ImGui::Button("Load Preset", buttonSize)) {
                if (HRPD_sim) {
                    HRPD_sim->m_OpManager.m_presetIndex = curPresetIndex;
                    HRPD_sim->m_OpManager.SetupPreset(&g_Viewer);
                }
            }
        }
    }

    { // Debug Store Data
        if (ImGui::Button("Store Data", { (w - p) * 0.5f, 0 })) {
            viewer->core().is_animating = false;
            auto hrpd = dynamic_cast<HRPDSimulator*>(m_sim.get());
            if (hrpd) {
                // hrpd->Debug_StoreData();
            }
        }
    }

    { // Buffer Mapping
        auto hrpd = dynamic_cast<HRPDSimulator*>(m_sim.get());
        if (hrpd) {
            bool _pre = g_InteractState.isBufferMapping;
            ImGui::Checkbox("Buffer Mapping", &g_InteractState.isBufferMapping);
            if (_pre != g_InteractState.isBufferMapping) {
                ;
            }
        }
    }

    { // Fullspace Collison Handling
        auto hrpd = dynamic_cast<HRPDSimulator*>(m_sim.get());
        if (hrpd) {
            bool _pre = g_InteractState.isUseFullspaceVerticesForCollisionHandling;
            ImGui::Checkbox("Fullspace Collison Handling", &g_InteractState.isUseFullspaceVerticesForCollisionHandling);
            if (_pre != g_InteractState.isUseFullspaceVerticesForCollisionHandling) {
                ;
            }
        }
    }

    ImGui::End();
}

static float RenderProfilerSection(const Util::Profiler::Section& sec, float x, float y, float full_width, float full_width_time, const std::function<float(const Util::Profiler::Section&)>& timefunc)
{
    const float LINE_HEIGHT = 16;

    double s_time = timefunc(sec);
    double s_width = (s_time / full_width_time) * full_width;
    auto rgba = std::hash<std::string>()(sec.name) * 256;
    ImU32 col = ImGui::GetColorU32({ ((rgba >> 24) & 0xFF) / 255.0f, ((rgba >> 16) & 0xFF) / 255.0f, ((rgba >> 8) & 0xFF) / 255.0f, 1.0f });

    ImVec2 min = { x, y };
    ImVec2 max = { static_cast<float>(min.x + s_width), min.y + LINE_HEIGHT };
    ImGui::RenderFrame(min, max, col);
    ImGui::RenderText(min, sec.name.c_str());

    if (s_width > 180) {
        std::string str = std::to_string(s_time * 1000.0) + std::string("ms@") + std::to_string(sec._numExec);
        ImVec2 text_size = ImGui::CalcTextSize(str.c_str());
        ImGui::RenderText({ max.x - text_size.x, min.y }, str.c_str());
    }

    float dx = 0;
    for (const Util::Profiler::Section& sub_sec : sec.sections) {
        dx += RenderProfilerSection(sub_sec, x + dx, y + LINE_HEIGHT, s_width, s_time, timefunc);
    }

    if (ImGui::IsWindowFocused() && ImGui::IsMouseHoveringRect(min, max)) {
        ImGui::SetTooltip("%s\n"
                          "\n"
                          "last %fms\n"
                          "las %fms\n"
                          "sum %fms\n"
                          "exc %u\n"
                          "%%p  %f",
            sec.name.c_str(),
            sec._lastTime * 1000.0f,
            sec._lastTime * 1000.0f,
            sec._sumTime * 1000.0f,
            (uint32_t)sec._numExec,
            (float)(sec.parent ? sec._sumTime / sec.parent->_sumTime : std::numeric_limits<float>::quiet_NaN()));
    }

    return s_width;
}
void PDViewer::ProfilerWindow()
{
    ImGui::Begin("Profiler");
    static int s_SelectedProfilerIdx = 0;
    static std::pair<const char*, Util::Profiler*> PROFILERS[] = {
        { "Simulator PreCompute", &g_PreComputeProfiler },
        { "Simulator Step", &g_StepProfiler },
        { "Frame", &g_FrameProfiler },
    };
    Util::Profiler& prof = *PROFILERS[s_SelectedProfilerIdx].second;

    static int s_SelectedTimeFunc = 0;
    static std::pair<const char*, std::function<float(const Util::Profiler::Section& sec)>> s_TimeFuncs[] = {
        { "LastTime", [](const Util::Profiler::Section& sec) { return sec._lastTime; } },
        { "SumTime", [](const Util::Profiler::Section& sec) { return sec._sumTime; } },
    };

    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("###Profiler", PROFILERS[s_SelectedProfilerIdx].first)) {
        for (int i = 0; i < std::size(PROFILERS); ++i) {
            if (ImGui::Selectable(PROFILERS[i].first, s_SelectedProfilerIdx == i)) {
                s_SelectedProfilerIdx = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    if (ImGui::BeginCombo("###ProfilerTimeFunc", s_TimeFuncs[s_SelectedTimeFunc].first)) {
        for (int i = 0; i < std::size(s_TimeFuncs); ++i) {
            if (ImGui::Selectable(s_TimeFuncs[i].first, s_SelectedTimeFunc == i)) {
                s_SelectedTimeFunc = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();

    if (ImGui::Button("Reset")) {
        prof.laterClearRootSection();
    }

    { // Profiler Section
        const Util::Profiler::Section& sec = prof.GetRootSection();
        auto& timefunc = s_TimeFuncs[s_SelectedTimeFunc].second;
        ImVec2 begin = { ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMin().x, ImGui::GetWindowPos().y + ImGui::GetWindowContentRegionMin().y + 20 };
        float width = ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x;
        RenderProfilerSection(sec, begin.x, begin.y, width, timefunc(sec), timefunc);
    }
    ImGui::End();

    // reset step profiler each frame
    if (viewer->core().is_animating == true) {
        g_StepProfiler.laterClearRootSection();
    }
}

// ====================================================================================================
// ---------------- Utils ----------------
// ====================================================================================================
void PDViewer::ExportToPNG(std::string export_url)
{
    bool offlineMode = false;
    if (offlineMode) return;

    double scale = 1;
    viewer->data().point_size *= scale;

    int width = static_cast<int>(scale * (viewer->core().viewport[2] - viewer->core().viewport[0]));
    int height = static_cast<int>(scale * (viewer->core().viewport[3] - viewer->core().viewport[1]));

    // Allocate temporary buffers for image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

    // Draw the scene in the buffers
    viewer->core().draw_buffer(viewer->data(), false, R, G, B, A);

    // Save it to a PNG
    igl::png::writePNG(R, G, B, A, export_url);

    viewer->data().point_size /= scale;
}

}