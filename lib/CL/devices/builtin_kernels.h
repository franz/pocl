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

typedef enum {
  // CD = custom device, BI = built-in
  POCL_CDBI_COPY_I8 = 0,
  POCL_CDBI_ADD_I32 = 1,
  POCL_CDBI_MUL_I32 = 2,
  POCL_CDBI_LEDBLINK = 3,
  POCL_CDBI_COUNTRED = 4,
  POCL_CDBI_DNN_CONV2D_RELU_I8 = 5,
  POCL_CDBI_SGEMM_LOCAL_F32 = 6,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32_SCALE = 7,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32 = 8,
  POCL_CDBI_ABS_F32 = 9,
  POCL_CDBI_DNN_DENSE_RELU_I8 = 10,
  POCL_CDBI_MAXPOOL_I8 = 11,
  POCL_CDBI_ADD_I8 = 12,
  POCL_CDBI_MUL_I8 = 13,
  POCL_CDBI_ADD_I16 = 14,
  POCL_CDBI_MUL_I16 = 15,
  POCL_CDBI_STREAMOUT_I32 = 16,
  POCL_CDBI_STREAMIN_I32 = 17,
  POCL_CDBI_VOTE_U32 = 18,
  POCL_CDBI_VOTE_U8 = 19,
  POCL_CDBI_DNN_CONV2D_NCHW_F32 = 20,
  POCL_CDBI_OPENVX_SCALEIMAGE_NN_U8 = 21,
  POCL_CDBI_OPENVX_SCALEIMAGE_BL_U8 = 22,
  POCL_CDBI_OPENVX_TENSORCONVERTDEPTH_WRAP_U8_F32 = 23,
  POCL_CDBI_OPENVX_MINMAXLOC_R1_U8 = 24,
  POCL_CDBI_SOBEL3X3_U8 = 25,
  POCL_CDBI_PHASE_U8 = 26,
  POCL_CDBI_MAGNITUDE_U16 = 27,
  POCL_CDBI_ORIENTED_NONMAX_U16 = 28,
  POCL_CDBI_CANNY_U8 = 29,
  POCL_CDBI_STREAM_MM2S_P512 = 30,
  POCL_CDBI_STREAM_S2MM_P512 = 31,
  POCL_CDBI_BROADCAST_1TO2_P512 = 32,
  POCL_CDBI_SOBEL3X3_P512 = 33,
  POCL_CDBI_PHASE_P512 = 34,
  POCL_CDBI_MAGNITUDE_P512 = 35,
  POCL_CDBI_ORIENTED_NONMAX_P512 = 36,
  POCL_CDBI_GAUSSIAN3X3_P512 = 37,
  POCL_CDBI_DBK_KHR_GEMM = 38,
  POCL_CDBI_DBK_KHR_MATMUL = 39,
  POCL_CDBI_LAST,
  POCL_CDBI_JIT_COMPILER = 0xFFFF
} BuiltinKernelId;

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

#ifdef __cplusplus
}
#endif


#endif
