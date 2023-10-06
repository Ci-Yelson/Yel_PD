add_rules("mode.debug", "mode.release")

add_requires("spdlog", "glfw", "glad", "glew", "glm", "nlohmann_json")
add_requires("tetgen 1.6.0", {alias = "tetgen"})
add_requires("libigl v2.4.0", {alias = "libigl", system = false, configs = {imgui = false}})

add_requires("amgcl 1.4.3", {alias = "amgcl"}) -- deps on `boost`
-- add_requires("boost")


option("PD_USE_CUDA")
    set_default(true) -- 设置默认值为 true
    set_showmenu(true)
    set_description("The PD_USE_CUDA config option")
option_end()

target("Yel_PD")
    set_kind("binary")
    
    set_languages("cxx17")
    add_defines("IGL_VIEWER_VIEWER_QUIET")
    -- add_defines("EIGEN_DONT_PARALLELIZE")
    
    add_defines("POLYSOLVE_WITH_AMGCL")

    set_options("PD_USE_CUDA")

    add_rules("c++")
    -- add_files("src/**.tpp") -- error: unknown source file: src\polysolve\LinearSolverEigen.tpp
    add_files(
        -- 递归添加src下的所有cpp文件
        "src/**.cpp", 

        -- imgui
        "libs/imgui/imgui.cpp",
        "libs/imgui/imgui_draw.cpp",
        "libs/imgui/imgui_widgets.cpp",
        "libs/imgui/imgui_tables.cpp",
        "libs/imgui/imgui_demo.cpp",
        "libs/imgui/backends/imgui_impl_glfw.cpp",
        "libs/imgui/backends/imgui_impl_opengl3.cpp",
        -- imguizmo
        "libs/imguizmo/ImGuizmo.cpp"
    )
    add_includedirs(
        "src", 
        "libs",
        "libs/imgui",
        "libs/imguizmo",
        "libs/CppNumericalSolvers/include",
        "libs/LBFGSpp/include"
    )

    add_packages("spdlog", "glfw", "glad", "glew", "glm", "nlohmann_json")
    add_packages("tetgen", "libigl")

    add_packages("amgcl")

    -- For cuda
    if has_config("PD_USE_CUDA") then
        add_files("src/**.cu")
        add_cugencodes("native")
        -- add_cugencodes("compute_61")
        -- add_cugencodes("compute_75")
        -- 链接 CUDA 库
        add_links("cudart", "cublas", "cusparse")
        -- 添加编译定义
        add_defines("PD_USE_CUDA")
    end


-- Config for windows: xmake f -p windows -a x64 -m release --cc=clang-cl --cxx=clang-cl
-- Config for commands_json: xmake project -k compile_commands
-- Check cuda: xmake l detect.sdks.find_cuda

--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro defination
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

