#include <iostream>

#include "Simulator/HRPD/HRPDSimulator.hpp"
#include "Util/Profiler.hpp"
#include "Viewer/PDViewer.hpp"
#include "igl/opengl/ViewerData.h"
#include "igl/opengl/glfw/Viewer.h"

#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

#include <memory>
#include <spdlog/spdlog.h>

UI::InteractState g_InteractState;

Util::Profiler g_FrameProfiler;
Util::Profiler g_StepProfiler;
Util::Profiler g_PreComputeProfiler;
igl::opengl::glfw::Viewer g_Viewer;
igl::opengl::glfw::imgui::ImGuizmoWidget g_Gizmo;

int main(int argc, char** argv)
{
#ifndef EIGEN_DONT_PARALLELIZE
    Eigen::setNbThreads(PD_EIGEN_NUM_THREADS);
    Eigen::initParallel();
    spdlog::info("Eigen::nbThreads() = {}", Eigen::nbThreads());
#endif

    if (argc < 2) {
        std::cout << "USAGE: [.EXE] [CONFIGJSON]" << std::endl;
        return -1;
    }
    spdlog::set_level(spdlog::level::info);
    std::string configJSON_URl = argv[1];
    g_InteractState.load(configJSON_URl);

    std::shared_ptr<PD::PDViewer> viewerPlugin = std::make_shared<PD::PDViewer>();
    g_Viewer.plugins.push_back(viewerPlugin.get());

    g_Viewer.core().is_animating = false;
    g_Viewer.core().animation_max_fps = 30;
    g_Viewer.core().background_color << 100. / 255, 100. / 255, 100. / 255, 1.0;

    g_Viewer.launch_init(true, false, "PD Viewer");

    {
        g_Gizmo.init(&g_Viewer, nullptr);
        // g_Gizmo config
        g_Gizmo.visible = false; // TODO [###]
        g_Gizmo.operation = ImGuizmo::TRANSLATE;
        g_Gizmo.callback = [&](const Eigen::Matrix4f& T) {
            ImGuiIO& io = ImGui::GetIO();
            // ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
            spdlog::info(">>> ImGui::GetIO() - DisplaySize = ({}, {})", io.DisplaySize.x, io.DisplaySize.y);
            auto HRPD_sim = dynamic_cast<PD::HRPDSimulator*>(viewerPlugin->m_sim.get());
            if (!HRPD_sim || HRPD_sim->m_OpManager.m_selectObjectIndex == -1)
                return;
            auto& obj = HRPD_sim->m_OpManager.m_operationObjects[HRPD_sim->m_OpManager.m_selectObjectIndex];
            auto& T0 = obj->m_gizmoT.cast<double>();
            const Eigen::Matrix4d TT = (T.cast<double>() * T0.inverse()).transpose();
            obj->ApplyTransform(TT, g_Gizmo.T);
            g_Viewer.data(obj->m_meshID).set_vertices(obj->m_verts);
            g_Viewer.data(obj->m_meshID).compute_normals();
        };
    }

    viewerPlugin->Setup();
    g_Viewer.core().align_camera_center(viewerPlugin->m_sim->GetPositions().cast<double>(), viewerPlugin->m_sim->GetTriangles().cast<int>());

    g_Viewer.launch_rendering();
    g_Viewer.launch_shut();

    return 0;
}