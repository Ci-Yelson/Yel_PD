#include <iostream>

#include "Util/Profiler.hpp"
#include "Viewer/PDViewer.hpp"
#include "igl/opengl/ViewerData.h"
#include "igl/opengl/glfw/Viewer.h"

#include "igl/read_triangle_mesh.h"
#include <igl/remove_duplicate_vertices.h>

#include <memory>
#include <spdlog/spdlog.h>

UI::InteractState g_InteractState;

Util::Profiler g_FrameProfiler;
Util::Profiler g_StepProfiler;
Util::Profiler g_PreComputeProfiler;
igl::opengl::glfw::Viewer g_Viewer;

int main(int argc, char** argv)
{
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

    viewerPlugin->Setup();
    g_Viewer.core().align_camera_center(viewerPlugin->m_sim->GetPositions().cast<double>(), viewerPlugin->m_sim->GetTriangles().cast<int>());

    g_Viewer.launch_rendering();
    g_Viewer.launch_shut();

    return 0;
}