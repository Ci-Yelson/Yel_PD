#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/ViewerPlugin.h>
#include "Simulator/HRPD/HRPDSimulator.hpp"
#include "Simulator/PDSimulator.hpp"
#include "igl/unproject_onto_mesh.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <spdlog/spdlog.h>

#include "ImGuiContext/ImGuiContext.hpp"
#include "ImGuiContext/OpenGLFrameBuffer.hpp"

#include "Util/Profiler.hpp"

#include <memory>
#include <string>

namespace PD {

struct PDViewer : public igl::opengl::glfw::ViewerPlugin {
    std::unique_ptr<OpenGLFrameBuffer> m_frameBuffer;
    std::unique_ptr<ImGuiContext> m_imguiContext;

    std::shared_ptr<PDSimulator> m_sim;

    bool isSceneInterationActive = true;
    // std::pair<int, int> m_sceneWindowSize = { 1280, 800 };
    std::pair<int, int> m_sceneWindowSize = { -1, -1 };
    ImVec2 m_sceneWindowPos;
    ImVec2 m_sceneCursorPos;

    Util::StopWatch m_frameTimer;
    Util::StopWatch m_predrawTimer;
    Util::StopWatch m_drawTimer;
    Util::StopWatch m_postdrawTimer;

    // For floor
    int m_floorMeshID = -1;
    float m_floorGridSize = 50.0f;

    // For interact, i.g. dragging
    int m_interactMeshID = -1;

public:
    PDViewer()
    {
        if (g_InteractState.simulatorType == "HRPD") {
            spdlog::info("### USE HRPD SIMULATOR");
            auto tetMesh = std::make_shared<HRPDTetMesh>(g_InteractState.meshURL);
            m_sim = std::make_shared<HRPDSimulator>(tetMesh);
        }
        else if (g_InteractState.simulatorType == "QNPD") {
            spdlog::info("### USE QNPD SIMULATOR [TODO]");
            // auto tetMesh = std::make_shared<QNPDTetMesh>(g_InteractState.meshURL);
            // m_sim = std::make_shared<QNPDSimulator>(tetMesh);
        }
    }

    ~PDViewer()
    {
        m_imguiContext->end();
    }

    // Config gravity and floor
    void Setup();
    void Reset();
    
public:
    void init(igl::opengl::glfw::Viewer* _viewer) override
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
    void shutdown() override
    {
        m_imguiContext->end();
    }

    bool pre_draw() override;

    bool mouse_down(int button, int modifier) override;
    bool mouse_move(int mouse_x, int mouse_y) override;
    bool mouse_up(int button, int modifier) override;
    bool mouse_scroll(float delta_y) override { return !isSceneInterationActive; }

    bool key_pressed(unsigned int key, int modifiers) override { return !isSceneInterationActive; }
    bool key_down(int key, int modifiers) override { return !isSceneInterationActive; }
    bool key_up(int key, int modifiers) override { return !isSceneInterationActive; }

private:
    void SimulatorInfoWindow();
    void OperationWindow();
    void ProfilerWindow();
};
}; // namespace PD