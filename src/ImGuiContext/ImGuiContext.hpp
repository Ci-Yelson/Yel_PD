#pragma once

#include "GLFW/glfw3.h"
#include "imgui.h"

namespace PD {
struct ImGuiContext {
    ImVec2 root_window_pos;

    bool init(GLFWwindow* window);
    void pre_render();
    void post_render();
    void end();
};
} // namespace PD