if(TARGET igl::core)
    return()
endif()

message(STATUS "Third-party: creating target 'igl::core'")

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(libigl)