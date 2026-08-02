# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)

project(world-explorer
	VERSION 1.0
	DESCRIPTION "WorldFoundry native world exploration interface"
	LANGUAGES C CXX CUDA
)
set(NGP_VERSION "${CMAKE_PROJECT_VERSION}")

if (NOT NGP_DEPLOY)
	set(NGP_VERSION "${NGP_VERSION}dev")
endif()

option(NGP_BUILD_WITH_GUI "Build with GUI support (requires GLFW and GLEW)?" ON)
option(NGP_BUILD_WITH_PYTHON_BINDINGS "Build Python bindings for World Explorer?" ON)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/CMakeLists.txt")
	message(FATAL_ERROR
		"Some World Explorer dependencies are missing. "
		"Run: python -m worldfoundry.studio.native.world_explorer setup"
	)
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    set(CMAKE_CUDA_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

# Figure out CUDA version
if(CMAKE_CUDA_COMPILER_LOADED)
	if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
		set(CUDA_VERSION "${CMAKE_MATCH_1}")
	endif()
endif()

# Adapted from the CMake source code at https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
# Simplified to return a semicolon-separated list of the compute capabilities of installed devices
function(TCNN_AUTODETECT_CUDA_ARCHITECTURES OUT_VARIABLE)
	if (NOT TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		if (CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
			set(file "${PROJECT_BINARY_DIR}/detect_tcnn_cuda_architectures.cu")
		else()
			set(file "${PROJECT_BINARY_DIR}/detect_tcnn_cuda_architectures.cpp")
		endif()

		file(WRITE ${file} ""
			"#include <cuda_runtime.h>\n"
			"#include <cstdio>\n"
			"int main() {\n"
			"	int count = 0;\n"
			"	if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
			"	if (count == 0) return -1;\n"
			"	for (int device = 0; device < count; ++device) {\n"
			"		cudaDeviceProp prop;\n"
			"		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
			"			std::printf(\"%d%d\", prop.major, prop.minor);\n"
			"			if (device < count - 1) std::printf(\";\");\n"
			"		}\n"
			"	}\n"
			"	return 0;\n"
			"}\n"
		)

		try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file} RUN_OUTPUT_VARIABLE compute_capabilities)
		if (run_result EQUAL 0)
			# If the user has multiple GPUs with the same compute capability installed, list that capability only once.
			list(REMOVE_DUPLICATES compute_capabilities)
			set(TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT ${compute_capabilities} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
		endif()
	endif()

	if (NOT TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		message(STATUS "Automatic GPU detection failed. Building for Turing and Ampere as a best guess.")
		set(${OUT_VARIABLE} "75;86" PARENT_SCOPE)
	else()
		set(${OUT_VARIABLE} ${TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT} PARENT_SCOPE)
	endif()
endfunction()

set(TCNN_CUDA_ARCHITECTURES "" CACHE STRING "Build tiny-cuda-nn for a specific GPU architecture.")
if (DEFINED ENV{TCNN_CUDA_ARCHITECTURES})
	message(STATUS "Obtained CUDA architectures from environment variable TCNN_CUDA_ARCHITECTURES=$ENV{TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{TCNN_CUDA_ARCHITECTURES})
elseif (TCNN_CUDA_ARCHITECTURES)
	message(STATUS "Obtained CUDA architectures from CMake variable TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
else()
	message(STATUS "Obtained CUDA architectures automatically from installed GPUs")
	TCNN_AUTODETECT_CUDA_ARCHITECTURES(CMAKE_CUDA_ARCHITECTURES)
endif()

# If the CUDA version does not support the chosen architecture, target
# the latest supported one instead.
if (CUDA_VERSION VERSION_LESS 11.0)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 75)
elseif (CUDA_VERSION VERSION_LESS 11.1)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 80)
elseif (CUDA_VERSION VERSION_LESS 11.8)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 86)
else()
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 50)
else()
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 20)
endif()

foreach (CUDA_CC IN LISTS CMAKE_CUDA_ARCHITECTURES)
	if (CUDA_CC GREATER ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(WARNING "CUDA version ${CUDA_VERSION} is too low for detected architecture ${CUDA_CC}. Targeting the highest supported architecture ${LATEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()

	if (CUDA_CC LESS ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(ERROR "CUDA version ${CUDA_VERSION} no longer supports detected architecture ${CUDA_CC}. Targeting the lowest supported architecture ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()
endforeach(CUDA_CC)

if (NOT CMAKE_CUDA_ARCHITECTURES)
	list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

message(STATUS "Targeting CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.")
endif()

list(APPEND NGP_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})
list(APPEND NGP_LIBRARIES cuda)

###############################################################################
# Dependencies
###############################################################################

add_subdirectory("dependencies/fmt")
list(APPEND NGP_LIBRARIES fmt)
list(APPEND NGP_INCLUDES "dependencies/fmt/include")

if (NGP_BUILD_WITH_GUI)
	# OpenGL
	find_package(OpenGL REQUIRED)
	# GLFW (as object library) needs X11-related link deps explicitly on Linux.
	if (UNIX AND NOT APPLE)
		find_package(X11 REQUIRED)
	endif()

	# GLFW
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
	set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

	add_subdirectory(dependencies/glfw)

	set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

	mark_as_advanced(
		GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
		GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
		GLFW_USE_OSMESA GLFW_USE_RETINA GLFW_USE_MIR
		BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
	)

	list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/glfw/include" "dependencies/imgui")

	if (MSVC)
		list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/gl3w")
		list(APPEND GUI_SOURCES "dependencies/gl3w/GL/gl3w.c")
		list(APPEND NGP_LIBRARIES opengl32 glfw)
	else()
		find_package(GLEW REQUIRED)
		list(APPEND NGP_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
		list(APPEND NGP_LIBRARIES GL ${GLEW_LIBRARIES} glfw)
		if (UNIX AND NOT APPLE)
			# Required for some GLFW/X11 code paths (e.g. XRRSetCrtcGamma via Xrandr).
			list(APPEND NGP_LIBRARIES
				${X11_LIBRARIES}
				${X11_Xrandr_LIB}
				${X11_Xi_LIB}
				${X11_Xcursor_LIB}
				${X11_Xinerama_LIB}
			)
		endif()
	endif()

	list(APPEND GUI_SOURCES
		dependencies/imguizmo/ImGuizmo.cpp
		dependencies/imgui/imgui.cpp
		dependencies/imgui/backends/imgui_impl_glfw.cpp
		dependencies/imgui/backends/imgui_impl_opengl3.cpp
		dependencies/imgui/imgui_draw.cpp
		dependencies/imgui/imgui_tables.cpp
		dependencies/imgui/imgui_widgets.cpp
		dependencies/imgui/misc/cpp/imgui_stdlib.cpp
	)

	list(APPEND NGP_DEFINITIONS -DNGP_GUI)
endif(NGP_BUILD_WITH_GUI)

list(APPEND NGP_INCLUDE_DIRECTORIES
	"dependencies"
	"dependencies/filesystem"
	"dependencies/tinylogger"
)

find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (NGP_BUILD_WITH_PYTHON_BINDINGS)
	find_package(Python 3.7 COMPONENTS Interpreter Development)
	if (Python_FOUND)
		# Python bindings need pybind11. The vendored pybind11 in this repo is too old for Python 3.11+,
		# so prefer a system-installed pybind11 (e.g. `pip install pybind11`) in that case.
		if (Python_VERSION VERSION_GREATER_EQUAL "3.11")
			execute_process(
				COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
				OUTPUT_VARIABLE _PYBIND11_CMAKE_DIR
				OUTPUT_STRIP_TRAILING_WHITESPACE
				ERROR_QUIET
			)
			if (_PYBIND11_CMAKE_DIR)
				list(APPEND CMAKE_PREFIX_PATH "${_PYBIND11_CMAKE_DIR}")
			endif()

			find_package(pybind11 CONFIG QUIET)
			if (NOT pybind11_FOUND)
				message(WARNING
					"Python ${Python_VERSION} detected, but the vendored pybind11 is too old for Python 3.11+. "
					"Could not find a system pybind11 CMake package. Disabling NGP_BUILD_WITH_PYTHON_BINDINGS. "
					"Fix by running `pip install pybind11` (in the same Python environment) or use Python<=3.10."
				)
				set(NGP_BUILD_WITH_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings for World Explorer?" FORCE)
				set(Python_FOUND FALSE)
			endif()
		else()
			add_subdirectory("dependencies/pybind11")
		endif()
	endif()
endif()


###############################################################################
# Program
###############################################################################

list(APPEND NGP_DEFINITIONS -DNGP_VERSION="${NGP_VERSION}")
list(APPEND NGP_INCLUDE_DIRECTORIES "include")
if (NOT MSVC)
	list(APPEND NGP_LIBRARIES ${CMAKE_DL_LIBS})
endif()
list(APPEND NGP_SOURCES
	${GUI_SOURCES}
	src/tiny-cuda-nn/common_host.cu
	src/camera_path.cu
	src/common_host.cu
	src/render_buffer.cu
	src/testbed.cu
	src/thread_pool.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")

add_library(ngp STATIC ${NGP_SOURCES})
set_target_properties(ngp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(ngp PUBLIC ${NGP_DEFINITIONS})
target_compile_options(ngp PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(ngp PUBLIC ${NGP_INCLUDE_DIRECTORIES})
target_link_directories(ngp PUBLIC ${NGP_LINK_DIRECTORIES})
target_link_libraries(ngp PUBLIC ${NGP_LIBRARIES})

if (MSVC)
	file(GLOB CUDA_DLLS "${CUDA_COMPILER_BIN}/cudart64*.dll")
	if (CUDA_DLLS)
		add_custom_command(TARGET ngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CUDA_DLLS} "${CMAKE_CURRENT_BINARY_DIR}" COMMAND_EXPAND_LISTS)
	endif()
endif()

if (Python_FOUND)
	add_library(pyngp SHARED src/python_api.cu)
	set_target_properties(pyngp PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
	target_link_libraries(pyngp PRIVATE ngp fmt PUBLIC ${PYTHON_LIBRARIES} pybind11::module)
	target_compile_definitions(pyngp PUBLIC -DNGP_PYTHON)
	pybind11_extension(pyngp)

	# Make `import pyngp` work when running the native viewer from the source tree.
	if (MSVC)
		add_custom_command(TARGET pyngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pyngp> "${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:pyngp>")
	else()
		add_custom_command(TARGET pyngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pyngp> "${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:pyngp>")
	endif()
endif()
