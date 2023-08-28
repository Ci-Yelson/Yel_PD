cmake_minimum_required(VERSION 3.22.1)
project(Yel_PD LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
# For clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Use ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif()

##################################################################
set(PD_USE_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
##################################################################

if (PD_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 61;75)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
add_compile_definitions(EIGEN_DONT_PARALLELIZE)

add_executable(${PROJECT_NAME})
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*)
target_sources(${PROJECT_NAME} PUBLIC ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

# Add cmake/
list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# libigl
find_package(libigl REQUIRED)
igl_include(glfw)
igl_include(copyleft tetgen)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl_copyleft::tetgen)
target_compile_definitions(${PROJECT_NAME} PUBLIC IGL_VIEWER_VIEWER_QUIET)
# Add libs/ : imgui_docking, imguizmo
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/libs)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/libs/imgui)
target_sources(${PROJECT_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR}/libs/imgui/imgui.cpp
    ${PROJECT_SOURCE_DIR}/libs/imgui/imgui_draw.cpp
    ${PROJECT_SOURCE_DIR}/libs/imgui/imgui_widgets.cpp
    ${PROJECT_SOURCE_DIR}/libs/imgui/imgui_tables.cpp
    ${PROJECT_SOURCE_DIR}/libs/imgui/imgui_demo.cpp
    ${PROJECT_SOURCE_DIR}/libs/imgui/backends/imgui_impl_glfw.cpp
    ${PROJECT_SOURCE_DIR}/libs/imgui/backends/imgui_impl_opengl3.cpp

    ${PROJECT_SOURCE_DIR}/libs/imguizmo/ImGuizmo.cpp
)
# OpenMP
find_package(OpenMP REQUIRED)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
# spdlog
find_package(spdlog REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog_header_only)
# json
find_package(nlohmann_json REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)

# CUDA
if (PD_USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PUBLIC PD_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cu)
    # target_sources(${PROJECT_NAME} PUBLIC ${SRC_FILES})
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda-12.2/include)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
endif()