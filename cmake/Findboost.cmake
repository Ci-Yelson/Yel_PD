#
# Copyright 2021 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
if(TARGET Boost::boost)
    return()
endif()

message(STATUS "Third-party: creating targets 'Boost::boost'...")

set(PREVIOUS_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(OLD_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 使用 SourceForge 作为备用下载源（如果 JFrog 不可用）
set(BOOST_URL "https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.bz2/download" CACHE STRING "Boost download URL")
set(BOOST_URL_SHA256 "6478edfe2f3305127cffe8caf73ea0176c53769f4bf1585be237eb30798c3b8e" CACHE STRING "Boost download URL SHA256 checksum")

include(FetchContent)

# 下载 Boost 源码
FetchContent_Declare(
    boost
    URL ${BOOST_URL}
    URL_HASH SHA256=${BOOST_URL_SHA256}
)

FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    FetchContent_Populate(boost)
    set(BOOST_SOURCE ${boost_SOURCE_DIR})
    set(Boost_POPULATED ON)
endif()

# Only build the following Boost libs
set(BOOST_LIBS_OPTIONAL "" CACHE STRING "Boost libs to be compiled" FORCE)

# File lcid.cpp from Boost_locale.cpp doesn't compile on MSVC, so we exclude them from the default
# targets being built by the project (only targets explicitly used by other targets will be built).
# 下载 boost-cmake
FetchContent_Declare(
    boost-cmake
    GIT_REPOSITORY https://github.com/Orphis/boost-cmake.git
    GIT_TAG 7f97a08b64bd5d2e53e932ddf80c40544cf45edf
)

FetchContent_GetProperties(boost-cmake)
if(NOT boost-cmake_POPULATED)
    FetchContent_Populate(boost-cmake)
    add_subdirectory(${boost-cmake_SOURCE_DIR} ${boost-cmake_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ${OLD_CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_CXX_FLAGS "${PREVIOUS_CMAKE_CXX_FLAGS}")

foreach(name IN ITEMS
        atomic
        chrono
        container
        date_time
        filesystem
        iostreams
        log
        system
        thread
        timer
    )
    if(TARGET Boost_${name})
        set_target_properties(Boost_${name} PROPERTIES FOLDER third_party/boost)
    endif()
endforeach()