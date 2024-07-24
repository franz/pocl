/*******************************************************************************
 * Copyright (c) 2022-2024 Henry Linjamäki, Michal Babej / Intel Finland Oy
 *
 * PoCL-specific proof-of-concept (draft) of Defined Builtin Kernels extension.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#ifndef OPENCL_EXP_DEFINED_BUILTIN_KERNELS
#define OPENCL_EXP_DEFINED_BUILTIN_KERNELS

#include "cl_exp_tensor.h"

/* errors returned by the DBK API */
#define CL_INVALID_DBK_ID 0x8101
#define CL_INVALID_DBK_ATTRIBUTE 0x8102
#define CL_INVALID_DBK_RANK 0x8103
#define CL_INVALID_DBK_SHAPE 0x8104
#define CL_INVALID_DBK_DATATYPE 0x8105

/* TODO numeric values */
#define CL_INVALID_TENSOR_LAYOUT -2309
#define CL_INVALID_TENSOR_RANK -2310
#define CL_INVALID_TENSOR_SHAPE -2311
#define CL_UNSUPPORTED_DBK -2312


typedef enum
{
  /* CD = custom device, BI = built-in */
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

/* for storing cl_dbk_property enums and actual values */
typedef cl_properties cl_dbk_properties;

typedef enum
{
  /* Maximum relative error in ULPs allowed for the results respect to */
  /* infinitely precise result. */
  CL_DBK_PROPERTY_MAX_RELATIVE_ERROR = 1, /* <float> */

  /* allow the N-th tensor argument to be mutable with respect to dimensions */
  CL_DBK_PROPERTY_MUTABLE_TENSOR_DIMS, /* <cl_uint> */

  /* allow the N-th tensor argument to be mutable with respect to data types */
  CL_DBK_PROPERTY_MUTABLE_TENSOR_DTYPES, /* <cl_uint> */

  /* allow the N-th tensor argument to be mutable with respect to layout */
  CL_DBK_PROPERTY_MUTABLE_TENSOR_LAYOUT, /* <cl_uint> */

  /* Allows the results of the DBK to fluctuate* with the exactly same
   * inputs across kernel launches.
   *
   * *: CL_DBK_PROPERTY_MAX_RELATIVE_ERROR must still be respected if present.
   *
   * Drivers may ignore this property. */
  CL_DBK_PROPERTY_NON_DETERMINISTIC,

  /* Allow driver to trade off accuracy for speed by allowing it to flush
   * denormals to zero.
   *
   * Drivers may ignore this property, meaning the behavior is not guaranteed. */
  CL_DBK_PROPERTY_ALLOW_FTZ
} cl_dbk_property;

typedef cl_program (*clCreateProgramWithDefinedBuiltInKernels_fn) (
    cl_context context, cl_uint num_devices, const cl_device_id *device_list,
    cl_uint num_kernels, const BuiltinKernelId *kernel_ids, const char **kernel_names,
    const void **kernel_attributes, cl_int *device_support, cl_int *errcode_ret);

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithDefinedBuiltInKernels (cl_context context,
                                          cl_uint num_devices,
                                          const cl_device_id *device_list,
                                          cl_uint num_kernels,
                                          const BuiltinKernelId *kernel_ids,
                                          const char **kernel_names,
                                          const void **kernel_attributes,
                                          cl_int *device_support,
                                          cl_int *errcode_ret);

#define CL_MAX_DBK_PROPERTIES 64

/* Name: "khr_gemm"
 * Attributes for General multiply operation for matrices.
 *
 * Note that this DBK can also perform matrix-vector operations if
 * tensor shapes are set accordingly. */
typedef struct _cl_dbk_attributes_khr_gemm
{
  cl_tensor_desc a, b, c_in, c_out;
  cl_bool trans_a, trans_b;
  /* Union, real Type depends on the tensor operands. E.g.
   * CL_TENSOR_FLOAT --> cl_float, CL_TENSOR_DOUBLE --> cl_double. */
  cl_tensor_datatype_union alpha, beta;
  /* 0-terminated array */
  cl_dbk_properties kernel_props[CL_MAX_DBK_PROPERTIES];
} cl_dbk_attributes_khr_gemm;

/* Name: "khr_matmul"
 * Attributes for Matrix multiplication. Identical to khr_gemm
 * with alpha and beta set to 1 and 0, respectively.
 *
 * Note that this DBK can also perform matrix-vector operations if
 * tensor shapes are set accordingly. */
typedef struct _cl_dbk_attributes_khr_matmul
{
  cl_tensor_desc a, b, c;
  cl_int trans_a;
  cl_int trans_b;
  /* 0-terminated array */
  cl_dbk_properties kernel_props[CL_MAX_DBK_PROPERTIES];
} cl_dbk_attributes_khr_matmul;

#endif /* OPENCL_EXP_DEFINED_BUILTIN_KERNELS */
