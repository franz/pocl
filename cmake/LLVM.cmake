
#=============================================================================
#   CMake build system files for detecting Clang and LLVM
#
#   Copyright (c) 2014-2020 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

if(DEFINED LLVM_DIR OR DEFINED ENV{LLVM_DIR} OR DEFINED LLVM_DIR)
  # user provided CMake package root, use it
  find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH)
elseif(DEFINED WITH_LLVM_CONFIG AND WITH_LLVM_CONFIG)
  # user provided llvm-config, use the preferred version
  if(IS_ABSOLUTE "${WITH_LLVM_CONFIG}")
    if(EXISTS "${WITH_LLVM_CONFIG}")
      set(LLVM_CONFIG "${WITH_LLVM_CONFIG}")
    endif()
  else()
    find_program(LLVM_CONFIG NAMES "${WITH_LLVM_CONFIG}")
  endif()
else()
  # search for LLVMConfig.cmake of supported versions in descending order
  find_package(LLVM 19.1.0...<19.2 CONFIG)
  if(NOT LLVM_FOUND)
    find_package(LLVM 18.1.0...<18.2 CONFIG)
  endif()
  if(NOT LLVM_FOUND)
    find_package(LLVM 17.0.0...<17.1 CONFIG)
  endif()
  if(NOT LLVM_FOUND)
    find_package(LLVM 16.0.0...<16.1 CONFIG)
  endif()
  if(NOT LLVM_FOUND)
    find_package(LLVM 15.0.0...<15.1 CONFIG)
  endif()
  if(NOT LLVM_FOUND)
    find_package(LLVM 14.0.0...<14.1 CONFIG)
  endif()
  # at last, fallback to finding any llvm-config executable
  if(NOT LLVM_FOUND)
  find_program(LLVM_CONFIG
    NAMES
      "llvmtce-config"
      "llvm-config"
      "llvm-config-mp-19.0" "llvm-config-mp-19" "llvm-config-19" "llvm-config190"
      "llvm-config-mp-18.0" "llvm-config-mp-18" "llvm-config-18" "llvm-config180"
      "llvm-config-mp-17.0" "llvm-config-mp-17" "llvm-config-17" "llvm-config170"
      "llvm-config-mp-16.0" "llvm-config-mp-16" "llvm-config-16" "llvm-config160"
      "llvm-config-mp-15.0" "llvm-config-mp-15" "llvm-config-15" "llvm-config150"
      "llvm-config-mp-14.0" "llvm-config-mp-14" "llvm-config-14" "llvm-config140"
      "llvm-config"
    DOC "llvm-config executable")
  endif()
endif()

# if we have the llvm-config only, get the cmake dir
if(LLVM_CONFIG AND NOT LLVM_FOUND)
  # A macro to run llvm config
  macro(run_llvm_config VARIABLE_NAME)
    execute_process(
      COMMAND "${LLVM_CONFIG}" ${ARGN}
      OUTPUT_VARIABLE LLVM_CONFIG_VALUE
      RESULT_VARIABLE LLVM_CONFIG_RETVAL
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(LLVM_CONFIG_RETVAL)
      message(SEND_ERROR "Error running llvm-config with arguments: ${ARGN}")
    else()
      set(${VARIABLE_NAME} ${LLVM_CONFIG_VALUE} CACHE STRING "llvm-config's ${VARIABLE_NAME} value")
      message(STATUS "llvm-config's ${VARIABLE_NAME} is: ${${VARIABLE_NAME}}")
    endif()
  endmacro(run_llvm_config)
  run_llvm_config(LLVM_CMAKEDIR --cmakedir)
  find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH HINTS "${LLVM_CMAKEDIR}")
endif()

if(NOT LLVM_CONFIG AND NOT LLVM_FOUND)
  message(FATAL_ERROR "Could not find either llvm-config or LLVMConfig.cmake !")
endif()

if((LLVM_VERSION_MAJOR LESS 14) OR (LLVM_VERSION_MAJOR GREATER 19))
  message(FATAL_ERROR "LLVM version between 14.0 and 19.0 required, found: ${LLVM_VERSION}")
endif()

#####################################################

message(STATUS "LLVM CMAKE dir: ${LLVM_CMAKE_DIR}")

# LLVM_CMAKE_DIR contains LLVM_INSTALL_PREFIX/lib/cmake/llvm
get_filename_component(CLANG_CMAKE_DIR "${LLVM_CMAKE_DIR}" DIRECTORY)
set(CLANG_CMAKE_DIR "${CLANG_CMAKE_DIR}/clang")
message(STATUS "Clang CMAKE dir: ${CLANG_CMAKE_DIR}")

find_package(Clang CONFIG REQUIRED HINTS "${CLANG_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddClang)

# check the host compiler is good enough to compile LLVM
include(CheckCompilerVersion)

# check if atomics are supported with/without extra LD flags
include(CheckAtomic)

############

include(LLVMPoclSetup)
