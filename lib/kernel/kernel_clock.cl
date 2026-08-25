#include "templates.h"

#if !defined(__has_builtin)
#error "__has_builtin is not available"
#endif

static ulong __pocl_read_cpu_clock() {
#if (defined(__x86_64__) || defined(__i386__))

#if __has_builtin(__builtin_readcyclecounter)
  return __builtin_readcyclecounter();
#else
#error "on X86-64 / i386, and __builtin_readcyclecounter is not available"
#endif

#elif defined(__arm__) || defined(__aarch64__) || defined(__riscv)
#if __has_builtin(__builtin_readsteadycounter)
  return __builtin_readsteadycounter();
#else
#error "on RISCV / ARM, and __builtin_readsteadycounter is not available"
#endif

#else
#error "unknown architecture, dont know which builtin to use for timing"
#endif

}

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_device(void) {
  return __pocl_read_cpu_clock();
}

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_work_group(void) {
  return __pocl_read_cpu_clock();
}

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_sub_group(void) {
  return __pocl_read_cpu_clock();
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_device(void) {
  ulong clk = __pocl_read_cpu_clock();
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_work_group(void) {
  ulong clk = __pocl_read_cpu_clock();
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_sub_group(void) {
  ulong clk = __pocl_read_cpu_clock();
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}
