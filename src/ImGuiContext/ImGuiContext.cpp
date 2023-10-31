#include "ImGuiContext/ImGuiContext.hpp"

#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <spdlog/spdlog.h>

namespace PD {

inline void SetupImGuiStyle(bool bStyleDark_, float alpha_)
{
    // clang-format off
    ImGuiStyle& style = ImGui::GetStyle();

    if( bStyleDark_ )
    {
        ImGui::StyleColorsDark();
    }
    else
    {
        ImGui::StyleColorsLight();
    }

    for (int i = 0; i <= ImGuiCol_COUNT; i++)
    {
        ImGuiCol_ ei = (ImGuiCol_)i;
        ImVec4& col = style.Colors[i];
        if(  (ImGuiCol_ModalWindowDimBg  != ei ) &&
            ( ImGuiCol_NavWindowingDimBg != ei ) &&
            ( col.w < 1.00f || ( ImGuiCol_FrameBg  == ei )
                            || ( ImGuiCol_WindowBg == ei ) 
                            || ( ImGuiCol_ChildBg  == ei ) ) )
        {
            col.w = alpha_ * col.w;
        }
    }
    
    style.ChildBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.PopupBorderSize = 1.0f;
    style.WindowBorderSize = 0.0f;
    style.FrameRounding = 3.0f;
    style.Alpha = 1.0f;
    // clang-format on
}

bool ImGuiContext::init(GLFWwindow* window)
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows

    // {
    //     // clang-format off
    //     auto& colors = ImGui::GetStyle().Colors;
    //     colors[ImGuiCol_WindowBg] = ImVec4{ 0.1f, 0.1f, 0.1f, 1.0f };

    //     colors[ImGuiCol_Header] = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };
    //     colors[ImGuiCol_HeaderHovered] = ImVec4{ 0.3f, 0.3f, 0.3f, 1.0f };
    //     colors[ImGuiCol_HeaderActive] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };

    //     colors[ImGuiCol_Button] = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };
    //     colors[ImGuiCol_ButtonHovered] = ImVec4{ 0.3f, 0.3f, 0.3f, 1.0f };
    //     colors[ImGuiCol_ButtonActive] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };

    //     colors[ImGuiCol_FrameBg] = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };
    //     colors[ImGuiCol_FrameBgHovered] = ImVec4{ 0.3f, 0.3f, 0.3f, 1.0f };
    //     colors[ImGuiCol_FrameBgActive] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };

    //     colors[ImGuiCol_Tab] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
    //     colors[ImGuiCol_TabHovered] = ImVec4{ 0.38f, 0.38f, 0.38f, 1.0f };
    //     colors[ImGuiCol_TabActive] = ImVec4{ 0.28f, 0.28f, 0.28f, 1.0f };
    //     colors[ImGuiCol_TabUnfocused] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
    //     colors[ImGuiCol_TabUnfocusedActive] = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };

    //     colors[ImGuiCol_TitleBg] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
    //     colors[ImGuiCol_TitleBgActive] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
    //     colors[ImGuiCol_TitleBgCollapsed] = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };

    //     ImGuiStyle& style = ImGui::GetStyle();
    //     if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    //         style.WindowRounding = 0.0f;
    //         style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    //     }
    //     // clang-format on
    // }

    SetupImGuiStyle(false, 1.0f);

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    spdlog::info(">>> ImGuiContext::init - ok");

    return true;
}

void ImGuiContext::pre_render()
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Create the docking environment
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("InvisibleWindow", nullptr, windowFlags);
    ImGui::PopStyleVar(3);

    root_window_pos = ImGui::GetWindowPos();

    ImGuiID dockspace_id = ImGui::GetID("InvisibleWindowDockSpace");

    if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr) {
        // Clear out existing layout
        ImGui::DockBuilderRemoveNode(dockspace_id);
        // Add empty node
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
        // Main node should cover entire window
        ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetWindowSize());
        // Build dock layout
        ImGuiID dock_id_center = dockspace_id;
        ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Down, 0.20f, nullptr, &dock_id_center);
        ImGuiID dock_id_Lift = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Left, 0.20f, nullptr, &dock_id_center);
        ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Right, 0.20f, nullptr, &dock_id_center);

        // dock windows ...
        ImGui::DockBuilderDockWindow("Scene", dock_id_center);
        ImGui::DockBuilderDockWindow("SimulatorInfo", dock_id_Lift);
        ImGui::DockBuilderDockWindow("Operation", dock_id_right);
        ImGui::DockBuilderDockWindow("Profiler", dock_id_bottom);

        ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    // spdlog::info(">>> ImGuiContext pre_render - ok");
}

void ImGuiContext::post_render()
{
    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    ImGuiIO& io = ImGui::GetIO();

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }

    // spdlog::info(">>> ImGuiContext post_render - ok");
}

void ImGuiContext::end()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

}