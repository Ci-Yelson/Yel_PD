#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/ViewerPlugin.h>

#include "Simulator/HRPD/HRPDSimulator.hpp"
#include "Simulator/Newton/NewtonSimulator.hpp"

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
    // For OpenGL context
    std::unique_ptr<OpenGLFrameBuffer> m_frameBuffer;
    std::unique_ptr<ImGuiContext> m_imguiContext;

    bool isSceneInteractionActive = true;
    std::pair<int, int> m_sceneWindowSize = { 1280, 800 };
    ImVec2 m_sceneWindowPos;
    ImVec2 m_sceneCursorPos;

    // For Simulation
    std::shared_ptr<PDSimulator> m_sim;

    // Utils
    Util::StopWatch m_frameTimer;
    Util::StopWatch m_predrawTimer;
    Util::StopWatch m_drawTimer;
    Util::StopWatch m_postdrawTimer;

    // For floor
    int m_floorMeshID = -1;
    float m_floorGridSize = 1.0f;

    // For interact, i.g. draw dragging points or any other interaction infos
    int m_interactMeshID = -1;

public:
    // Init simulator
    PDViewer();
    // ~PDViewer() { m_imguiContext->end(); }

    // Config floor Mesh and interact Mesh.
    void Setup();
    // Stop animating and Reset simulator.
    void Reset();

public: // -- override --
    // Call when g_Viewer.launch_init()
    void init(igl::opengl::glfw::Viewer* _viewer) override;
    // Call when g_Viewer.launch_shut()
    void shutdown() override { m_imguiContext->end(); }

    // Draw total frame - Note: return true;
    // Call when g_Viewer.launch_rendering()
    bool pre_draw() override;

    // User interaction
    bool mouse_down(int button, int modifier) override;
    bool mouse_move(int mouse_x, int mouse_y) override;
    bool mouse_up(int button, int modifier) override;
    bool mouse_scroll(float delta_y) override { return !isSceneInteractionActive; }

    bool key_pressed(unsigned int key, int modifiers) override { return !isSceneInteractionActive; }
    bool key_down(int key, int modifiers) override { return !isSceneInteractionActive; }
    bool key_up(int key, int modifiers) override { return !isSceneInteractionActive; }

private: // -- windows --
    void SimulatorInfoWindow();
    void OperationWindow();
    void ProfilerWindow();

public: // -- utils --
    void ExportToPNG(std::string export_url);
    // void ExportToGIF(std::string export_url); //todo
};

}; // namespace PD