
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

#run_llvm_config(LLVM_BINDIR --bindir)
#string(REPLACE "${LLVM_INSTALL_PREFIX}" "${LLVM_INSTALL_PREFIX_CMAKE}" LLVM_BINDIR "${LLVM_BINDIR}")
#run_llvm_config(LLVM_LIBDIR --libdir)
#string(REPLACE "${LLVM_INSTALL_PREFIX}" "${LLVM_INSTALL_PREFIX_CMAKE}" LLVM_LIBDIR "${LLVM_LIBDIR}")
#run_llvm_config(LLVM_INCLUDEDIR --includedir)
#string(REPLACE "${LLVM_INSTALL_PREFIX}" "${LLVM_INSTALL_PREFIX_CMAKE}" LLVM_INCLUDEDIR "${LLVM_INCLUDEDIR}")
set(LLVM_BINARY_SUFFIX "-${LLVM_VERSION_MAJOR}")
set(LLVM_BINDIR "${LLVM_TOOLS_BINARY_DIR}")
set(LLVM_LIBDIR "${LLVM_LIBRARY_DIR}")
# LLVM_INCLUDE_DIR

#run_llvm_config(LLVM_ALL_TARGETS --targets-built)
#run_llvm_config(LLVM_HOST_TARGET --host-target)
#run_llvm_config(LLVM_BUILD_MODE --build-mode)
#run_llvm_config(LLVM_ASSERTS_BUILD --assertion-mode)
set(LLVM_TARGETS_BUILT ${LLVM_TARGETS_TO_BUILD})
set(LLVM_HOST_TARGET ${LLVM_HOST_TRIPLE})
set(LLVM_BUILD_MODE ${LLVM_BUILD_TYPE})
set(LLVM_ASSERTS_BUILD ${LLVM_ENABLE_ASSERTIONS})
set(LLVM_HAS_RTTI "${LLVM_ENABLE_RTTI}")

set(LLC_TRIPLE ${LLVM_TARGET_TRIPLE})
if(NOT LLC_TRIPLE)
  # fallback for older LLVM
  set(LLC_TRIPLE ${TARGET_TRIPLE})
endif()
if(NOT LLC_TRIPLE)
  message(FATAL_ERROR "LLC triple unset: ${LLVM_TARGET_TRIPLE}")
endif()

###########################################################################

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

if(LLVM_BUILD_MODE MATCHES "Debug")
  set(LLVM_BUILD_MODE_DEBUG 1)
else()
  set(LLVM_BUILD_MODE_DEBUG 0)
endif()

####################################################################

set(POCL_CLANG_COMPONENTS clangCodeGen clangFrontendTool
  clangFrontend clangDriver clangSerialization
  clangParse clangSema clangRewrite clangRewriteFrontend
  clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore clangAnalysis clangEdit
  clangAST clangASTMatchers clangLex clangBasic)

if(LLVM_VERSION_MAJOR GREATER 14)
   list(APPEND POCL_CLANG_COMPONENTS clangSupport)
endif()
# must come after clangFrontend
if(LLVM_VERSION_MAJOR GREATER 17)
   list(INSERT POCL_CLANG_COMPONENTS 4 clangAPINotes)
endif()

set(POCL_LLVM_COMPONENTS
  LLVMDemangle
  LLVMSupport
  LLVMCore
  LLVMCodeGen
  LLVMCodeGenTypes
  LLVMIRPrinter
  LLVMIRReader
  LLVMBitReader
  LLVMBitWriter
  LLVMBitstreamReader
  LLVMGlobalISel
  LLVMBinaryFormat
  LLVMTransformUtils
  LLVMInstrumentation
  LLVMInstCombine
  LLVMScalarOpts
  LLVMipo
  LLVMVectorize
  LLVMLinker
  LLVMAnalysis
  LLVMMC
  LLVMMCParser
  LLVMObjCopy
  LLVMObject
  LLVMOption
  LLVMRemarks
  LLVMDebugInfoDWARF
  LLVMExecutionEngine
  LLVMTarget
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Disassembler
  LLVMX86TargetMCA
  LLVMX86Desc
  LLVMX86Info
  LLVMPasses
  LLVMTargetParser
  LLVMLibDriver
  )

# LLVM_ENABLE_SHARED_LIBS = LLVM is built with shared component libraries
# (libLLVMxyz.so ); the same applies to libclangxyz.so

if(STATIC_LLVM)
  # link with static libLLVM
  if(LLVM_ENABLE_SHARED_LIBS)
    message(FATAL_ERROR "STATIC_LLVM=ON but LLVM built with shared libs only")
  endif()
  # libLLVM & libclang-cpp are always shared (AFAIK)
  set(LLVM_LIBS "${POCL_LLVM_COMPONENTS}")
  set(CLANG_LIBS "${POCL_CLANG_COMPONENTS}")
  set(LLVM_LINK_TYPE STATIC)
else()
  # link with shared libLLVM
  if(LLVM_ENABLE_SHARED_LIBS)
    set(LLVM_LIBS "${POCL_LLVM_COMPONENTS}")
    set(CLANG_LIBS "${POCL_CLANG_COMPONENTS}")
  else()
    # if shared component libs are disabled, link with single shared library
    set(LLVM_LIBS "LLVM")
    set(CLANG_LIBS "clang-cpp")
  endif()
  set(LLVM_LINK_TYPE SHARED)
endif()

set(POCL_CLANG_LINK_TARGETS ${CLANG_LIBS})
set(POCL_LLVM_LINK_TARGETS ${LLVM_LIBS})

####################################################################

macro(find_program_or_die OUTPUT_VAR PROG_NAME DOCSTRING)
  find_program(${OUTPUT_VAR}
    NAMES "${PROG_NAME}${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}"
    "${PROG_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    HINTS "${LLVM_BINDIR}"
    DOC "${DOCSTRING}"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
  )
  if(${OUTPUT_VAR})
    message(STATUS "Found ${PROG_NAME}: ${${OUTPUT_VAR}}")
  else()
    message(FATAL_ERROR "${PROG_NAME} executable not found!")
  endif()
endmacro()

find_program_or_die(CLANG "clang" "clang binary")
execute_process(COMMAND "${CLANG}" "--version" OUTPUT_VARIABLE LLVM_CLANG_VERSION RESULT_VARIABLE CLANG_RES)
find_program_or_die( CLANGXX "clang++" "clang++ binary")
execute_process(COMMAND "${CLANGXX}" "--version" OUTPUT_VARIABLE LLVM_CLANGXX_VERSION RESULT_VARIABLE CLANGXX_RES)
if(CLANGXX_RES OR CLANG_RES)
  message(FATAL_ERROR "Failed running clang/clang++ --version")
endif()

find_program_or_die(LLVM_OPT  "opt"       "LLVM optimizer")
find_program_or_die(LLVM_LLC  "llc"       "LLVM static compiler")
find_program_or_die(LLVM_AS   "llvm-as"   "LLVM assembler")
find_program_or_die(LLVM_LINK "llvm-link" "LLVM IR linker")
find_program_or_die(LLVM_LLI  "lli"       "LLVM interpreter")

if(ENABLE_LLVM_FILECHECKS)
  if(IS_ABSOLUTE "${LLVM_FILECHECK_BIN}" AND EXISTS "${LLVM_FILECHECK_BIN}")
    message(STATUS "LLVM IR checks enabled using ${LLVM_FILECHECK_BIN}.")
  else()
    find_program_or_die(LLVM_FILECHECK_BIN "FileCheck" "LLVM FileCheck (not installed by default)")
  endif()
endif()

if(NOT DEFINED LLVM_SPIRV)
  find_program(LLVM_SPIRV
    NAMES "llvm-spirv${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}"
    "llvm-spirv${CMAKE_EXECUTABLE_SUFFIX}"
    HINTS "${LLVM_BINDIR}" "${LLVM_INSTALL_PREFIX}")
  if(LLVM_SPIRV)
    execute_process(
        COMMAND "${LLVM_SPIRV}" "--version"
        OUTPUT_VARIABLE LLVM_SPIRV_VERSION_VALUE
        RESULT_VARIABLE LLVM_SPIRV_VERSION_RETVAL
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(${LLVM_SPIRV_VERSION_RETVAL} EQUAL 0)
        string(REGEX MATCH "LLVM version ([0-9]*)" LLVM_SPIRV_VERSION_MATCH ${LLVM_SPIRV_VERSION_VALUE})
        if(NOT ${CMAKE_MATCH_1} EQUAL ${LLVM_VERSION_MAJOR})
          message(WARNING "LLVM version of ${LLVM_SPIRV} does not \
            match LLVM's version ${LLVM_VERSION_MAJOR}; skipping" )
          unset(LLVM_SPIRV CACHE)
          unset(LLVM_SPIRV)
        endif()
    else()
      unset(LLVM_SPIRV)
    endif()
  endif()
  if(LLVM_SPIRV)
    set(HAVE_LLVM_SPIRV ON CACHE BOOL "" INTERNAL)
    message(STATUS "Found llvm-spirv: ${LLVM_SPIRV}")
  else()
    set(HAVE_LLVM_SPIRV OFF CACHE BOOL "" INTERNAL)
    message(STATUS "Did NOT find llvm-spirv!")
  endif()
endif()

if(NOT DEFINED SPIRV_LINK)
  find_program(SPIRV_LINK NAMES "spirv-link${CMAKE_EXECUTABLE_SUFFIX}"
    HINTS "${LLVM_BINDIR}" "${LLVM_INSTALL_PREFIX}")
  if(SPIRV_LINK)
    message(STATUS "Found spirv-link: ${SPIRV_LINK}")
  else()
    message(STATUS "Did NOT find spirv-link!")
  endif()
endif()

if(NOT DEFINED HAVE_LLVM_SPIRV_LIB)
  find_path(LLVM_SPIRV_INCLUDEDIR "LLVMSPIRVLib.h" PATHS "${LLVM_INCLUDE_DIR}/LLVMSPIRVLib" NO_DEFAULT_PATH)
  find_library(LLVM_SPIRV_LIB "LLVMSPIRVLib" PATHS "${LLVM_LIBDIR}" NO_DEFAULT_PATH)
  if(LLVM_SPIRV_INCLUDEDIR AND LLVM_SPIRV_LIB)
    set(HAVE_LLVM_SPIRV_LIB 1 CACHE BOOL "have LLVMSPIRVLib")
  else()
    set(HAVE_LLVM_SPIRV_LIB 0 CACHE BOOL "have LLVMSPIRVLib")
  endif()
endif()

####################################################################

# try compile with any compiler (supplied as argument)
macro(custom_try_compile_any SILENT COMPILER SUFFIX SOURCE RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.${SUFFIX}")
  file(WRITE "${RANDOM_FILENAME}" "${SOURCE}")

  math(EXPR LSIZE "${ARGC} - 4")

  execute_process(COMMAND "${COMPILER}" ${ARGN} "${RANDOM_FILENAME}" RESULT_VARIABLE RESV OUTPUT_VARIABLE OV ERROR_VARIABLE EV)
  if(${RESV} AND (NOT ${SILENT}))
    message(STATUS " ########## The command: ")
    string(REPLACE ";" " " ARGN_STR "${ARGN}")
    message(STATUS "${COMPILER} ${ARGN_STR} ${RANDOM_FILENAME}")
    message(STATUS " ########## Exited with nonzero status: ${${RES_VAR}}")
    if(OV)
      message(STATUS "STDOUT: ${OV}")
    endif()
    if(EV)
      message(STATUS "STDERR: ${EV}")
    endif()
  endif()
  file(REMOVE "${RANDOM_FILENAME}")

  set(${RES_VAR} ${RESV})
endmacro()

# convenience c/c++ source wrapper
macro(custom_try_compile_c_cxx COMPILER SUFFIX SOURCE1 SOURCE2 RES_VAR)
  set(SOURCE_PROG "
  ${SOURCE1}

  int main(int argc, char** argv) {

  ${SOURCE2}

  }")
  custom_try_compile_any(FALSE "${COMPILER}" ${SUFFIX} "${SOURCE_PROG}" ${RES_VAR} ${ARGN})
endmacro()

# convenience c/c++ source wrapper
macro(custom_try_compile_c_cxx_silent COMPILER SUFFIX SOURCE1 SOURCE2 RES_VAR)
  set(SOURCE_PROG "
  ${SOURCE1}

  int main(int argc, char** argv) {

  ${SOURCE2}

  }")
  custom_try_compile_any(TRUE "${COMPILER}" ${SUFFIX} "${SOURCE_PROG}" ${RES_VAR} ${ARGN})
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clangxx SOURCE1 SOURCE2 RES_VAR)
  custom_try_compile_c_cxx("${CLANGXX}" "cc" "${SOURCE1}" "${SOURCE2}" ${RES_VAR}  "-c" ${ARGN})
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clang SOURCE1 SOURCE2 RES_VAR)
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR}  "-c" ${ARGN})
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clang_silent SOURCE1 SOURCE2 RES_VAR)
  custom_try_compile_c_cxx_silent("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-c" ${ARGN})
endmacro()

# clang++ try-link macro
macro(custom_try_link_clang SOURCE1 SOURCE2 RES_VAR)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.${SUFFIX}")
  custom_try_compile_c_cxx_silent("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR}  "-o" "${RANDOM_FILENAME}" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

# clang try-compile-run macro, running via native executable
macro(custom_try_run_exe SOURCE1 SOURCE2 OUTPUT_VAR RES_VAR)
  set(OUTF "${CMAKE_BINARY_DIR}/try_run${CMAKE_EXECUTABLE_SUFFIX}")
  if(EXISTS "${OUTF}")
    file(REMOVE "${OUTF}")
  endif()
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" RESV "-o" "${OUTF}" "-x" "c")
  set(${OUTPUT_VAR} "")
  set(${RES_VAR} "")
  if(RESV OR (NOT EXISTS "${OUTF}"))
    message(STATUS " ########## Compilation failed")
  else()
    execute_process(COMMAND "${OUTF}" RESULT_VARIABLE RESV OUTPUT_VARIABLE ${OUTPUT_VAR} ERROR_VARIABLE EV)
    set(${RES_VAR} ${RESV})
    file(REMOVE "${OUTF}")
    if(${RESV})
      message(STATUS " ########## Running ${OUTF}")
      message(STATUS " ########## Exited with nonzero status: ${RESV}")
      if(${${OUTPUT_VAR}})
        message(STATUS " ########## STDOUT: ${${OUTPUT_VAR}}")
      endif()
      if(EV)
        message(STATUS " ########## STDERR: ${EV}")
      endif()
    endif()
  endif()
endmacro()

# clang try-compile-run macro, run via lli, the llvm interpreter
macro(custom_try_run_lli SILENT SOURCE1 SOURCE2 OUTPUT_VAR RES_VAR)
# this uses "lli" - the interpreter, so we can run any -target
# TODO variable for target !!
  set(OUTF "${CMAKE_BINARY_DIR}/try_run.bc")
  if(EXISTS "${OUTF}")
    file(REMOVE "${OUTF}")
  endif()
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" RESV "-o" "${OUTF}" "-x" "c" "-emit-llvm" "-c" ${ARGN})
  set(${OUTPUT_VAR} "")
  set(${RES_VAR} "")
  if(RESV OR (NOT EXISTS "${OUTF}"))
    message(STATUS " ########## Compilation failed")
  else()
    execute_process(COMMAND "${LLVM_LLI}" "-force-interpreter" "${OUTF}" RESULT_VARIABLE RESV OUTPUT_VARIABLE ${OUTPUT_VAR} ERROR_VARIABLE EV)
    set(${RES_VAR} ${RESV})
    file(REMOVE "${OUTF}")
    if(${RESV} AND (NOT ${SILENT}))
      message(STATUS " ########## The command ${LLVM_LLI} -force-interpreter ${OUTF}")
      message(STATUS " ########## Exited with nonzero status: ${RESV}")
      if(${${OUTPUT_VAR}})
        message(STATUS " ########## STDOUT: ${${OUTPUT_VAR}}")
      endif()
      if(EV)
        message(STATUS " ########## STDERR: ${EV}")
      endif()
    endif()
  endif()
endmacro()

####################################################################
####################################################################
####################################################################

macro(CHECK_ALIGNOF TYPE TYPEDEF OUT_VAR)

  if(NOT DEFINED "${OUT_VAR}")

    custom_try_run_lli(TRUE "
#ifndef offsetof
#define offsetof(type, member) ((char *) &((type *) 0)->member - (char *) 0)
#endif

${TYPEDEF}" "typedef struct { char x; ${TYPE} y; } ac__type_alignof_;
    int r = offsetof(ac__type_alignof_, y);
    return r;" SIZEOF_STDOUT RESULT "--target=${LLC_TRIPLE}")

    #message(FATAL_ERROR "SIZEOF: ${SIZEOF_STDOUT} RES: ${RESULT}")
    if(NOT ${RESULT})
      message(SEND_ERROR "Could not determine align of(${TYPE})")
      set(${OUT_VAR} "0" CACHE INTERNAL "Align of ${TYPE}")
    else()
      set(${OUT_VAR} "${RESULT}" CACHE INTERNAL "Align of ${TYPE}")
    endif()

  endif()

endmacro()

####################################################################

if(ENABLE_HOST_CPU_DEVICES)

# TODO: We need to set both target-triple and cpu-type when
# building, since the ABI depends on both. We can either add flags
# to all the scripts, or set the respective flags here in
# *_CLANG_FLAGS and *_LLC_FLAGS. Note that clang and llc use
# different option names to set these. Note that clang calls the
# triple "target" and the cpu "architecture", which is different
# from llc.

# Normalise the triple. Otherwise, clang normalises it when
# passing it to llc, which is then different from the triple we
# pass to llc. This would lead to inconsistent bytecode files,
# depending on whether they are generated via clang or directly
# via llc.

####################################################################

# FIXME: The cpu name printed by llc --version is the same cpu that will be
# targeted if you pass -mcpu=native to llc, so we could replace this auto-detection
# with just: set(LLC_HOST_CPU "native"), however, we can't do this at the moment
# because of the work-around for arm1176jz-s.
if(NOT DEFINED LLC_HOST_CPU_AUTO)
  message(STATUS "Find out LLC host CPU with ${LLVM_LLC}")
  execute_process(COMMAND ${LLVM_LLC} "--version" RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
  if(RES_VAR)
    message(FATAL_ERROR "Error ${RES_VAR} while determining LLC host CPU")
  endif()

  if(OUTPUT_VAR MATCHES "Host CPU: ([^ ]*)")
    # sigh... STRING(STRIP is to workaround regexp bug in cmake
    string(STRIP "${CMAKE_MATCH_1}" LLC_HOST_CPU_AUTO)
  else()
    message(FATAL_ERROR "Couldnt determine host CPU from llc output")
  endif()
endif()

if((LLC_HOST_CPU_AUTO MATCHES "unknown") AND (NOT LLC_HOST_CPU))
  message(FATAL_ERROR "LLVM could not recognize your CPU model automatically. Please run CMake with -DLLC_HOST_CPU=<cpu> (you can find valid names with: llc -mcpu=help)")
else()
  set(LLC_HOST_CPU_AUTO "${LLC_HOST_CPU_AUTO}" CACHE INTERNAL "Autodetected CPU")
endif()

if(DEFINED LLC_HOST_CPU)
  if(NOT LLC_HOST_CPU STREQUAL LLC_HOST_CPU_AUTO)
    message(STATUS "Autodetected CPU ${LLC_HOST_CPU_AUTO} overridden by user to ${LLC_HOST_CPU}")
  endif()
  set(SELECTED_HOST_CPU "${LLC_HOST_CPU}")
  set(HOST_CPU_FORCED 1 CACHE INTERNAL "CPU is forced by user")
else()
  set(SELECTED_HOST_CPU "${LLC_HOST_CPU_AUTO}")
  set(HOST_CPU_FORCED 0 CACHE INTERNAL "CPU is forced by user")
endif()

# Some architectures have -march and -mcpu reversed
if(NOT DEFINED CLANG_MARCH_FLAG)
  message(STATUS "Checking clang -march vs. -mcpu flag")
  custom_try_compile_clang_silent("" "return 0;" RES --target=${LLC_TRIPLE} -march=${SELECTED_HOST_CPU})
  if(NOT RES)
    set(CLANG_MARCH_FLAG "-march=")
  else()
    custom_try_compile_clang_silent("" "return 0;" RES --target=${LLC_TRIPLE} -mcpu=${SELECTED_HOST_CPU})
    if(NOT RES)
      set(CLANG_MARCH_FLAG "-mcpu=")
    else()
      message(FATAL_ERROR "Could not determine whether to use -march or -mcpu with clang")
    endif()
  endif()
  message(STATUS "  Using ${CLANG_MARCH_FLAG}")

  set(CLANG_MARCH_FLAG ${CLANG_MARCH_FLAG} CACHE INTERNAL "Clang option used to specify the target cpu")
endif()

endif(ENABLE_HOST_CPU_DEVICES)

####################################################################

# This tests that we can actually link to the llvm libraries.
# Mostly to catch issues like #295 - cannot find -ledit

#if(NOT LLVM_LINK_TEST)
#  message(STATUS "Running LLVM link test")
#  set(LLVM_LINK_TEST_FILENAME "${CMAKE_SOURCE_DIR}/cmake/LinkTestLLVM.cc")
#  try_compile(LLVM_LINK_TEST ${CMAKE_BINARY_DIR} "${LLVM_LINK_TEST_FILENAME}"
#              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
#              CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${LLVM_LIBDIR}"
#              LINK_LIBRARIES "${LLVM_LDFLAGS} ${LLVM_LIBS}"
#              COMPILE_DEFINITIONS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}"
#              OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT)
#  if (LLVM_LINK_TEST)
#    message(STATUS "LLVM link test OK")
#    set(LLVM_LINK_TEST 1 CACHE INTERNAL "LLVM link test result")
#  else()
#    message(STATUS "LLVM link test output: ${_TRY_COMPILE_OUTPUT}")
#    message(FATAL_ERROR "LLVM link test FAILED. This mostly happens when your LLVM installation does not have all dependencies installed.")
#  endif()
#endif()

####################################################################

# This tests that we can actually link to the Clang libraries.

#if(NOT CLANG_LINK_TEST)
#  message(STATUS "Running Clang link test")
#  set(CLANG_LINK_TEST_FILENAME "${CMAKE_SOURCE_DIR}/cmake/LinkTestClang.cc")
#  try_compile(CLANG_LINK_TEST ${CMAKE_BINARY_DIR} "${CLANG_LINK_TEST_FILENAME}"
#              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
#              CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${LLVM_LIBDIR}"
#              LINK_LIBRARIES "${LLVM_LDFLAGS} ${CLANG_LIBS} ${LLVM_LIBS} ${LLVM_SYSLIBS}"
#              COMPILE_DEFINITIONS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS} -DLLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}"
#              OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT)
#  if(CLANG_LINK_TEST)
#    message(STATUS "Clang link test OK")
#    set(CLANG_LINK_TEST 1 CACHE INTERNAL "Clang link test result")
#  else()
#    message(STATUS "Clang link test output: ${_TRY_COMPILE_OUTPUT}")
#    message(FATAL_ERROR "Clang link test FAILED. This mostly happens when your Clang installation does not have all dependencies and/or headers installed.")
#  endif()
#endif()


####################################################################

# Clang documentation on Language Extensions:
# __fp16 is supported on every target, as it is purely a storage format
# _Float16 is currently only supported on the following targets... SPIR, x86
# Limitations:
#     The _Float16 type requires SSE2 feature and above due to the instruction
#        limitations. When using it on i386 targets, you need to specify -msse2
#        explicitly.
#     For targets without F16C feature or above, please make sure:
#     Use GCC 12.0 and above if you are using libgcc.
#     If you are using compiler-rt, use the same version with the compiler.
#        Early versions provided FP16 builtins in a different ABI. A workaround is
#        to use a small code snippet to check the ABI if you cannot make sure of it.

if(ENABLE_HOST_CPU_DEVICES AND NOT DEFINED HOST_CPU_SUPPORTS_FLOAT16)
  set(HOST_CPU_SUPPORTS_FLOAT16 0)
  message(STATUS "Checking host support for _Float16 type")
    custom_try_compile_clang_silent("_Float16 callfp16(_Float16 a) { return a * 1.8f16; };" "_Float16 x=callfp16((_Float16)argc);"
    RESV --target=${LLC_TRIPLE} ${CLANG_MARCH_FLAG}${SELECTED_HOST_CPU})
  if(RESV EQUAL 0)
    set(HOST_CPU_SUPPORTS_FLOAT16 1)
  endif()
endif()

####################################################################

# TODO we should check double support of the target somehow (excluding emulation),
# for now just provide an option
if(ENABLE_HOST_CPU_DEVICES AND NOT DEFINED HOST_CPU_SUPPORTS_DOUBLE)
  if(X86)
    set(HOST_CPU_SUPPORTS_DOUBLE ON CACHE INTERNAL "FP64, always enabled on X86(-64)" FORCE)
  else()
    option(HOST_CPU_SUPPORTS_DOUBLE "Enable FP64 support for Host CPU device" ON)
  endif()
endif()

####################################################################

# to avoid creating an install target
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)
# neccesary hack. Unfontunately cmake targets for clang
# components have each hardcoded "LLVM" as dependency,
# therefore we cannot use our own determined LLVM dependencies,
# and the LLVMConfig.cmake as a whole is unusable (since it's not possible
# to link to purely static libraries of both clang & llvm).
set(POCL_LLVM_LINK_TARGETS "LLVM")
