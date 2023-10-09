if(TARGET spdlog)
	return()
endif()

message(STATUS "Third-party: creating target 'spdlog::spdlog'")

include(FetchContent)
FetchContent_Declare(
	spdlog
	GIT_REPOSITORY https://github.com/gabime/spdlog.git
	GIT_TAG v1.11.0
)
FetchContent_MakeAvailable(spdlog)