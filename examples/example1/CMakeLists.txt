#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
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

add_compile_options(${OPENCL_CFLAGS})

add_executable("example1" example1.c example1_exec.c example1.cl)
target_link_libraries("example1" ${POCLU_LINK_OPTIONS})
add_symlink_to_built_opencl_dynlib("example1")

add_test(NAME "examples/example1_dot_product" COMMAND "example1")

if (ENABLE_SPIRV)
  add_test(NAME "examples/example1_spirv" COMMAND "example1" "v")
  list(APPEND SPIREX "examples/example1_spirv")
endif()

set(PROPS)
if(NOT ENABLE_ANYSAN)
  set(PROPS
    PASS_REGULAR_EXPRESSION "[(]0[.]000000, 0[.]000000, 0[.]000000, 0[.]000000[)] [.] [(]0[.]000000, 0[.]000000, 0[.]000000, 0[.]000000[)] = 0[.]000000
[(]1[.]000000, 1[.]000000, 1[.]000000, 1[.]000000[)] [.] [(]1[.]000000, 1[.]000000, 1[.]000000, 1[.]000000[)] = 4[.]000000
[(]2[.]000000, 2[.]000000, 2[.]000000, 2[.]000000[)] [.] [(]2[.]000000, 2[.]000000, 2[.]000000, 2[.]000000[)] = 16[.]000000
[(]3[.]000000, 3[.]000000, 3[.]000000, 3[.]000000[)] [.] [(]3[.]000000, 3[.]000000, 3[.]000000, 3[.]000000[)] = 36[.]000000
OK")
endif()

set_tests_properties( "examples/example1_dot_product" ${SPIREX}
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    # devices which always support SPIR
    LABELS "level0"
    DEPENDS "pocl_version_check")

# devices which don't support SPIR
set_property(TEST "examples/example1_dot_product"
  APPEND PROPERTY LABELS "cpu;hsa;hsa-native;vulkan")

if(SPIREX AND HOST_CPU_ENABLE_SPIRV)
  set_property(TEST ${SPIREX} APPEND PROPERTY LABELS "cpu")
endif()

if (ENABLE_TCE)
  set_property(TEST "examples/example1_dot_product"
    APPEND PROPERTY LABELS "tce;almaif")
endif()

add_test(NAME "examples/example1_poclbin"
         COMMAND ${CMAKE_COMMAND}
            "-DCMD1=${CMAKE_BINARY_DIR}/bin/poclcc####-o####${CMAKE_BINARY_DIR}/example1.bin####${CMAKE_CURRENT_SOURCE_DIR}/example1.cl"
            "-DCMD2=${CMAKE_BINARY_DIR}/examples/example1/example1####b####${CMAKE_BINARY_DIR}/example1.bin"
            "-DCMD3=${CMAKE_COMMAND}####-E####remove####-f####${CMAKE_BINARY_DIR}/example1.bin"
            -P ${CMAKE_SOURCE_DIR}/cmake/multi_exec_test.cmake)

# Force disable the WG specialization to test executing from the generic
# binary only. This is to avoid the online compiler creating
# a specialized WG function that is ran instead of the binary we
# want to test here.
set_tests_properties( "examples/example1_poclbin"
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    ENVIRONMENT "POCL_KERNEL_CACHE=0;POCL_WORK_GROUP_SPECIALIZATION=0"
    LABELS "poclbin;cpu;level0"
    DEPENDS "pocl_version_check")

if(ENABLE_REMOTE_CLIENT AND ENABLE_REMOTE_SERVER AND ENABLE_HOST_CPU_DEVICES)
  add_test(NAME "remote/example1"
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh"
               "${CMAKE_BINARY_DIR}" "examples/example1/example1")
  set_tests_properties( "remote/example1"
    PROPERTIES
      COST 3.0
      ${PROPS}
      PROCESSORS 1
      LABELS "remote"
      DEPENDS "pocl_version_check"
      RESOURCE_LOCK "pocld_ports")

    if (ENABLE_SPIRV)
      add_test(NAME "remote/example1_spirv" COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scripts/test_remote_runner_single.sh" "${CMAKE_BINARY_DIR}" "examples/example1/example1" "v")

      set_tests_properties( "remote/example1_spirv"
        PROPERTIES
        COST 3.0
        ${PROPS}
        PROCESSORS 1
        LABELS "remote"
        DEPENDS "pocl_version_check"
        RESOURCE_LOCK "pocld_ports")

      list(APPEND SPIREX "remote/example1_spirv")
    endif()

endif()
