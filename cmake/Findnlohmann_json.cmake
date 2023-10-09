if(TARGET nlohmann_json)
	return()
endif()

message(STATUS "Third-party: creating target 'nlohmann_json::nlohmann_json'")


include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)