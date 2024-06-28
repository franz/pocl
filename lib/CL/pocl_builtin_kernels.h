#include "pocl_cl.h"
#include "pocl_export.h"

/*
  steps to add a new builtin kernel:

  1) add it to the end of BuiltinKernelId enum in this file, before POCL_CDBI_LAST

  2) open builtin_kernels.c and edit pocl_BIDescriptors, add a new struct
     for the new kernel, with argument metadata

  3) make sure that devices where you want to support this builtin kernel,
     report it. Every driver does this a bit differently, but at pocl_XYZ_init
     it must properly fill dev->builtin_kernel_list, dev->num_builtin_kernels
     Note: the kernel name reported to user should use dots as separators
     (example: pocl.add.apples.to.oranges)

  4) add the code for the builtin kernel for each device that will support it.
     Note: if the builtin kernel is in source format, its name in the source
     MUST have the dots replaced with underscore
     (example: pocl_add_apples_to_oranges)

     How to do this, depends on device:
       * CUDA has OpenCL-source builtins in lib/CL/devices/cuda/builtins.cl,
         it also has CUDA-source builtins in lib/CL/devices/cuda/builtins.cu
       * almaif driver with TTASIM backend has opencl-source builtins in
         lib/CL/devices/almaif/tce_builtins.cl
       * almaif driver with other backends has builtin kernels in binary format
  (bitstream)

*/

#ifndef POCL_BUILTIN_KERNELS_H
#define POCL_BUILTIN_KERNELS_H

#ifdef __cplusplus
extern "C"
{
#endif

#define BIKERNELS POCL_CDBI_LAST
POCL_EXPORT extern pocl_kernel_metadata_t pocl_BIDescriptors[BIKERNELS];

POCL_EXPORT
void pocl_init_builtin_kernel_metadata();

POCL_EXPORT
int pocl_setup_builtin_metadata(cl_device_id device, cl_program program,
                                unsigned program_device_i);

POCL_EXPORT
int pocl_sanitize_builtin_kernel_name(cl_kernel kernel, const char **saved_name);

POCL_EXPORT
int pocl_restore_builtin_kernel_name(cl_kernel kernel, const char *saved_name);

int pocl_validate_defined_builtin_attributes(BuiltinKernelId kernel_id,
                                            const void *kernel_attributes);

void *pocl_copy_defined_builtin_attributes(BuiltinKernelId kernel_id,
                                           const void *kernel_attributes);

int pocl_release_defined_builtin_attributes(BuiltinKernelId kernel_id,
                                            void *kernel_attributes);

#ifdef __cplusplus
}
#endif


#endif
