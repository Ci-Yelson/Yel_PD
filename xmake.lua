add_rules("mode.debug", "mode.release")

-- add_requires("spdlog", "glfw", "glad", "glew", "glm", "nlohmann_json")
-- add_requires("tetgen 1.6.0", {alias = "tetgen"})
-- add_requires("libigl v2.4.0", {alias = "libigl", system = false, configs = {imgui = false}})

-- add_requires("amgcl 1.4.3", {alias = "amgcl"}) -- deps on `boost`

add_requires("glfw", "glad", "glew", "glm")
add_requires("spdlog")
add_requires("nlohmann_json")

add_requires("openmp")
add_requires("stb")
add_requires("ghc_filesystem")
add_requires("boost")

-- libigl
package("libigl")
    set_homepage("https://libigl.github.io/")
    set_description("Simple C++ geometry processing library.")
    set_license("MPL-2.0")

    add_urls("https://github.com/libigl/libigl/archive/$(version).tar.gz",
             "https://github.com/libigl/libigl.git")
    add_versions("v2.2.0", "b336e548d718536956e2d6958a0624bc76d50be99039454072ecdc5cf1b2ede5")
    add_versions("v2.3.0", "9d6de3bdb9c1cfc61a0a0616fd96d14ef8465995600328c196fa672ee4579b70")
    add_versions("v2.4.0", "f3f53ee6f1e9a6c529378c6b0439cd2cfc0e30b2178b483fe6bea169ce6deb22")

    add_resources("2.x", "libigl_imgui", "https://github.com/libigl/libigl-imgui.git", "7e1053e750b0f4c129b046f4e455243cb7f804f3")

    add_configs("header_only", {description = "Use header only version.", default = true, type = "boolean"})
    add_configs("cgal", {description = "Use CGAL library.", default = false, type = "boolean"})
    add_configs("imgui", {description = "Use imgui with libigl.", default = false, type = "boolean"})

    if is_plat("windows") then
        add_syslinks("comdlg32")
    elseif is_plat("linux") then
        add_syslinks("pthread")
    end

    add_deps("cmake", "eigen")
    add_deps("tetgen 1.6.0", "stb")
    on_load("macosx", "linux", "windows", "mingw", function (package)
        if not package:config("header_only") then
            raise("Non-header-only version is not supported yet!")
        end
        if package:config("cgal") then
            package:add("deps", "cgal")
        end
        if package:config("imgui") then
            package:add("deps", "imgui", {configs = {glfw_opengl3 = true}})
        end
        package:add("deps", "stb")
    end)

    on_install("macosx", "linux", "windows", "mingw", function (package)
        if package:config("imgui") then
            local igl_imgui_dir = package:resourcefile("libigl_imgui")
            os.cp(path.join(igl_imgui_dir, "imgui_fonts_droid_sans.h"), package:installdir("include"))
        end
        if package:config("header_only") then
            os.cp("include/igl", package:installdir("include"))
            return
        end
        local configs = {"-DLIBIGL_BUILD_TESTS=OFF", "-DLIBIGL_BUILD_TUTORIALS=OFF", "-DLIBIGL_SKIP_DOWNLOAD=ON"}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or"OFF"))
        if not package:config("shared") then
            table.insert(configs, "-DLIBIGL_USE_STATIC_LIBRARY=ON")
        end
        if package:is_plat("windows") then
            table.insert(configs, "-DIGL_STATIC_RUNTIME=" .. (package:config("vs_runtime"):startswith("MT") and "ON" or "OFF"))
        end
        import("package.tools.cmake").install(package, configs)
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            void test() {
                Eigen::MatrixXd V(4,2);
                V<<0,0,
                   1,0,
                   1,1,
                   0,1;
                Eigen::MatrixXi F(2,3);
                F<<0,1,2,
                   0,2,3;
                Eigen::SparseMatrix<double> L;
                igl::cotmatrix(V,F,L);
            }
        ]]}, {configs = {languages = "c++14"}, includes = {"igl/cotmatrix.h", "Eigen/Dense", "Eigen/Sparse"}}))
    end)
package_end()

add_requires("libigl v2.4.0", {alias = "libigl", system = false})

-- mshio
package("mshio")
    -- set_kind("library")
    set_homepage("https://github.com/qnzhou/MshIO")
    set_description("A tiny C++ library to read/write ASCII/binary MSH format files.")
    set_license("Apache-2.0")

    add_urls("https://github.com/qnzhou/MshIO/archive/refs/tags/$(version).tar.gz",
            "https://github.com/qnzhou/MshIO.git")
    add_versions("v0.0.1", "ea3f6cb72a0d30e6af36a747be56ea5e60ed52b0ce10254e509a4e7b938714a3")

    add_deps("cmake")
    on_install(function (package) 
        local configs = {"-DLIBIGL_BUILD_TESTS=OFF", "-DLIBIGL_BUILD_TUTORIALS=OFF", "-DLIBIGL_SKIP_DOWNLOAD=ON"}
        import("package.tools.cmake").install(package, configs)
    end )

package_end()

add_requires("mshio v0.0.1", {alias = "mshio", system = false})


option("PD_USE_CUDA")
    -- set_default(true) -- 设置默认值为 true
    set_showmenu(true)
    set_description("The PD_USE_CUDA config option")
option_end()

target("Yel_PD")
    set_kind("binary")
    
    set_languages("cxx17")
    add_defines("IGL_VIEWER_VIEWER_QUIET")
    -- TODO
    add_defines("STB_IMAGE_IMPLEMENTATION")
    add_defines("STB_IMAGE_WRITE_IMPLEMENTATION")
    -- add_defines("EIGEN_DONT_PARALLELIZE")
    
    -- add_defines("POLYSOLVE_WITH_AMGCL")

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

    add_packages("glfw", "glad", "glew", "glm")
    add_packages("stb")

    add_packages("libigl")
    add_packages("mshio")
    add_packages("nlohmann_json")
    add_packages("spdlog")

    add_packages("ghc_filesystem")
    add_packages("boost")
    add_packages("mshio")

    add_packages("openmp")

    -- add_packages("amgcl")

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

