
#ifndef OPENCL_EXP_DEFINED_BUILTIN_KERNELS
#define OPENCL_EXP_DEFINED_BUILTIN_KERNELS

#include "cl_exp_tensor.h"

#define CL_INVALID_DBK_ID 0x8101
#define CL_INVALID_DBK_ATTRIBUTE 0x8102
#define CL_INVALID_DBK_RANK 0x8103
#define CL_INVALID_DBK_SHAPE 0x8104
#define CL_INVALID_DBK_DATATYPE 0x8105

typedef cl_properties cl_dbk_properties;

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


typedef enum
{
  // Maximum relative error in ULPs allowed for the results respect to
  // infinitely precise result.
  CL_DBK_PROPERTY_MAX_RELATIVE_ERROR = 1, // <float>

  // Built-in kernel attributes are immutable values (this allows
  // drivers to specialize their kernels). CL_DBK_MUTABLE_ATTR
  // followed by attribute index (cl_uint) enables the attribute to be
  // mutable via clSetKernelArg(attribute_index, ...).
  CL_DBK_PROPERTY_MUTABLE_ATTR, // <cl_uint>

  // Allows the results of the DBK to fluctuate* with the exactly same
  // inputs across kernel launches.
  //
  // *: CL_DBK_PROPERTY_MAX_RELATIVE_ERROR must still be respected if present.
  //
  // Drivers may ignore this property.
  CL_DBK_PROPERTY_NON_DETERMINISTIC,

  // Allow driver to trade off accuracy for speed by allowing it to flush
  // denormals to zero.
  //
  // Drivers may ignore this property, meaning the behavior is not guaranteed.
  CL_DBK_PROPERTY_ALLOW_FTZ
} cl_dbk_property;

typedef cl_kernel (CL_API_CALL *clCreateBuiltinKernelWithAttributesEXP_fn) (
    cl_program prog, const char *kernel_name, const void *kernel_attributes,
    cl_int *errcode_ret);

extern CL_API_ENTRY cl_kernel CL_API_CALL
clCreateBuiltinKernelWithAttributesEXP (cl_program prog,
                                        const char *kernel_name,
                                        const void *kernel_attributes,
                                        cl_int *errcode_ret);

// Name: "khr_gemm"
// General multiply operation for matrices.
//
// Note that this also performs matrix-vector operations by setting
// tensor shapes accordingly.
typedef struct _cl_dbk_attributes_khr_gemm
{
  const cl_tensor_desc *a;
  const cl_tensor_desc *b;
  const cl_tensor_desc *c_in;
  const cl_tensor_desc *c_out;
  cl_int trans_a;
  cl_int trans_b;
  // Pointers to scaler values. Type depends on the tensor operands. E.g.
  // CL_TENSOR_FLOAT --> cl_float, CL_TENSOR_DOUBLE --> cl_double.
  const void *alpha;
  const void *beta;
  const cl_dbk_properties *kernel_props;
} cl_dbk_attributes_khr_gemm;

// Name: "khr_matmul" Matrix multiplication. Alias for khr_gemm with
// alpha and beta set to 1 and 0, respectively
//
// Note that this also performs matrix-vector operations by setting
// tensor shapes accordingly.
typedef struct _cl_dbk_attributes_khr_matmul
{
  const cl_tensor_desc *a;
  const cl_tensor_desc *b;
  const cl_tensor_desc *c;
  cl_int trans_a;
  cl_int trans_b;
  const cl_dbk_properties *kernel_props;
} cl_dbk_attributes_khr_matmul;

#endif // OPENCL_EXP_DEFINED_BUILTIN_KERNELS
