/* OpenCL runtime library: clCreateBuiltinKernelWithAttributesEXP()

  Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

#include "pocl_cl.h"

#include <math.h>

#define MAX_KERNEL_NAME_LEN 1024

/*
public:
  TensorDescView(const cl_tensor_desc *TDesc) {
    if (!TDesc || !TDesc->shape)
      return;

    // TODO: assert the data layout is valid.

    OriginalDesc = TDesc;
  }

  bool valid() const noexcept { return OriginalDesc; }
  operator bool() const noexcept { return valid(); }

  size_t rank() const noexcept {
    assert(valid());
    return OriginalDesc->rank;
  }

  size_t operator[](size_t I) const noexcept {
    assert(valid());
    assert(I < rank());
    return OriginalDesc->shape[I];
  }

  cl_tensor_datatype dtype() const noexcept {
    assert(valid());
    return OriginalDesc->dtype;
  }

  bool shapeEquals(const TensorDescView &Other) {
    assert(valid());
    if (!Other.valid() || rank() != Other.rank())
      return false;

    for (unsigned I = 0, E = rank(); I < E; I++)
      if (operator[](I) != Other[I])
        return false;

    return true;
  }

*/

static cl_bool tensor_shape_equals(const cl_tensor_desc *A, const cl_tensor_desc *B) {
  assert(A);
  assert(B);
  if (A->rank != B->rank)
    return CL_FALSE;

  for (unsigned i = 0; i < A->rank; ++i) {
    if (A->shape[i] != B->shape[i])
      return CL_FALSE;
  }
  return CL_TRUE;
}

/*
  size_t numElements() const noexcept {
    assert(valid());
    size_t Result = 1;
    for (unsigned I = 0, E = rank(); I < E; I++)
      Result *= operator[](I);
    assert(Result);
    return Result;
  }
*/

static size_t tensor_num_elements(const cl_tensor_desc *A) {
  assert(A);
  size_t Result = 1;
  for (unsigned i = 0; i < A->rank; ++i) {
    Result *= A->shape[i];
  }
  return Result;
}

/*
  std::string toString() const {
    std::string Result = "(";
    for (unsigned I = 0, E = rank(); I < E; I++) {
      if (I != 0)
        Result += ", ";
      Result += std::to_string(operator[](I));
    }
    // TODO: dtype
    Result += ")";
    // TODO: layout
    return Result;
  }
*/

static void tensor_to_string(const cl_tensor_desc *A, char* String, cl_uint N) {
  assert(A);
  assert(N > 1);
  N = N - 1;
  String[0] = 0;
  strncat(String, "(", N);
  char IntegerStr[64];
  for (unsigned i = 0; i < A->rank; ++i) {
    if (i != 0)
      strncat(String, ",", N);
    IntegerStr[63] = 0;
    snprintf(IntegerStr, 63, "%zu", A->shape[i]);
    strncat(String, IntegerStr, N);
  }
  String[N-1] = 0;
}

/*
private:
  unsigned getTrailingDim(const cl_tensor_layout_blas &BlasDL) const {
    assert(valid());
    assert(rank() < sizeof(unsigned) * 8u &&
           "Too many dimensions for the bitset.");

    unsigned DimSet = (1u << rank()) - 1;
    for (unsigned I = 0; I < rank() - 1; I++)
      DimSet &= ~(1u << BlasDL.leading_dims[I]);

    assert(__builtin_popcount(DimSet) == 1 && "Invalid data layout?");
    unsigned TrailingDim = __builtin_ctz(DimSet);
    assert(TrailingDim < rank());
    return TrailingDim;
  }
};
*/

static unsigned tensor_get_trailing_dim(const cl_tensor_desc *A,
                                        const cl_tensor_layout_blas *BL) {
  assert(A);
  assert((A->rank < (sizeof(unsigned)*8)) &&
         "Too many dimensions for the bitset.");

  unsigned DimSet = (1u << A->rank) - 1;
  for (unsigned I = 0; I < A->rank - 1; I++)
    DimSet &= ~(1u << BL->leading_dims[I]);

  assert(__builtin_popcount(DimSet) == 1 && "Invalid data layout?");
  unsigned TrailingDim = __builtin_ctz(DimSet);
  assert(TrailingDim < A->rank);
  return TrailingDim;

}

/*
  // Get the stride for the 'Dim'th (zero-based) leading dimension,
  // measured in tensor elements. Applicable for 2D+ tensors with BLAS
  // datalayout.
  //
  // NthDim can be rank() - 1 or more in which case the result is
  // getBlasStrideInElts(rank - 2) * shape[trailing_dimension].
  size_t getBlasStrideInElts(unsigned Dim) const {
    assert(valid());
    assert(rank() >= 2);
    assert(OriginalDesc->layout && "Does not have data layout!");
    const auto *BaseDL =
        reinterpret_cast<const cl_tensor_layout_base *>(OriginalDesc->layout);
    assert(
        BaseDL->stype == CL_TENSOR_LAYOUT_BLAS &&
        "The method must not be called for tesnors with non-BLAS data layouts");

    const auto &BlasDL =
        *reinterpret_cast<const cl_tensor_layout_blas *>(BaseDL);

    if (Dim < rank() - 1)
      return BlasDL.leading_strides[Dim];

    return BlasDL.leading_strides[rank() - 1] * operator[](
                                                    getTrailingDim(BlasDL));
  }
*/

static size_t tensor_get_blas_stride_in_elements(const cl_tensor_desc *A,
                                                 unsigned Dim) {
  assert(A);
  assert(A->rank >= 2);
  assert(A->layout && "Does not have data layout!");
  assert(A->layout_type == CL_TENSOR_LAYOUT_BLAS &&
         "The method must not be called for tensors with non-BLAS data layouts");
  const cl_tensor_layout_blas *BL = (const cl_tensor_layout_blas *)A->layout;
  if (Dim < (A->rank - 1))
    return BL->leading_strides[Dim];
  else
    return BL->leading_strides[A->rank-1] * tensor_get_trailing_dim(A, BL);
}

/*
  bool isBlasRowMajor() const {
    assert(valid());
    assert(OriginalDesc->layout && "Does not have data layout!");
    const auto *BaseDL =
        reinterpret_cast<const cl_tensor_layout_base *>(OriginalDesc->layout);
    assert(
        BaseDL->stype == CL_TENSOR_LAYOUT_BLAS &&
        "The method must not be called for tesnors with non-BLAS data layouts");
    assert(rank() >= 2 && "Not a (batched) matrix!");

    const auto &BlasDL =
        *reinterpret_cast<const cl_tensor_layout_blas *>(BaseDL);
    return BlasDL.leading_dims[0] == (rank() - 1u);
  }
*/

static cl_bool tensor_is_blas_row_major(const cl_tensor_desc *A) {
  assert(A);
  assert(A->layout && "Does not have data layout!");
  assert(A->layout_type == CL_TENSOR_LAYOUT_BLAS &&
         "The method must not be called for tensors with non-BLAS data layouts");
  const cl_tensor_layout_blas *BL = (const cl_tensor_layout_blas *)A->layout;
  assert(A->rank >= 2 && "Not a (batched) matrix!");

  return BL->leading_dims[0] == (A->rank - 1u) ? CL_TRUE : CL_FALSE;
}

/*
static pocl_kernel_metadata_t *getKernelMetadata(cl_program Program,
                                                 const std::string KernelName) {
  for (size_t I = 0, E = Program->num_kernels; I != E; I++) {
    if (KernelName == Program->kernel_meta[I].name)
      return &Program->kernel_meta[I];
  }
  assert(!"Missing kernel metadata!");
}

static void runDBK(_cl_command_node *Cmd, void *Data) {
  auto *Runner = static_cast<std::function<void(_cl_command_node &)> *>(Data);
  (*Runner)(*Cmd);
}

static void releaseDBK(void *Data) {
  if (Data)
    delete static_cast<std::function<void(cl_kernel)> *>(Data);
}


template <typename PtrT>
static PtrT getBufferDataAs(const _cl_command_node &Cmd, unsigned ArgIdx) {
  const auto &Arg = Cmd.command.run.arguments[ArgIdx];
  if (Arg.is_raw_ptr)
    return static_cast<PtrT>(Arg.value);

  auto Mem = *static_cast<cl_mem *>(Arg.value);
  auto *Ptr =
      static_cast<char *>(Mem->device_ptrs[Cmd.device->global_mem_id].mem_ptr);
  Ptr += Arg.offset;
  return reinterpret_cast<PtrT>(Ptr);
}
*/

static cl_int createGemmKernel(cl_program Program, cl_bool TransposeA,
                               cl_bool TransposeB, const cl_tensor_desc *TenA,
                               const cl_tensor_desc *TenB, const cl_tensor_desc *TenCIOpt,
                               const cl_tensor_desc *TenCOut, const void *AlphaAttr,
                               const void *BetaAttr, char* OutInstanceName) {

  POCL_RETURN_ERROR_COND((TenA == NULL), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND((TenB == NULL), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND((TenCOut == NULL), CL_INVALID_DBK_ATTRIBUTE);

  // TBC: 4D+ tensor could be supported by treating the additional
  //      dimensions as batch dimensions - but it might not be
  //      worthwhile due the extra work to support them and processing
  //      overhead they may impose.
  POCL_RETURN_ERROR_ON((TenA->rank > 3), CL_INVALID_DBK_RANK,
                       "Unsupported high-degree tensors.");

  // TODO: Should we have something like CL_DBK_INVALID_TENSOR_SHAPE?
  POCL_RETURN_ERROR_COND((TenA->rank != TenB->rank), CL_INVALID_DBK_RANK);
  POCL_RETURN_ERROR_COND((TenB->rank != TenCOut->rank), CL_INVALID_DBK_RANK);

  POCL_RETURN_ERROR_ON((TenCIOpt != NULL && tensor_shape_equals(TenCIOpt, TenCOut) == CL_FALSE),
                       CL_INVALID_DBK_SHAPE, "Tensor shape mismatch between c_in and c_out.");

  // FIXME: check tensor shapes are correct respect to the transpose
  //        configurations.

  size_t BatchDims = TenA->rank - 2;

  // CO[b][m][n] = sigma_over_m_n_k(A[b][m][k] * B[b][k][n]) + CI[b][m][n].
  size_t Am = TenA->shape[BatchDims + 0];
  size_t Ak = TenA->shape[BatchDims + 1];
  size_t Bk = TenB->shape[BatchDims + 0];
  size_t Bn = TenB->shape[BatchDims + 1];
  size_t COm = TenCOut->shape[BatchDims + 0];
  size_t COn = TenCOut->shape[BatchDims + 1];

  // TODO: Should have more descriptive error code? Or would it be better
  //       to have error logging like the cl_program has for building?
  POCL_RETURN_ERROR_COND((Ak != Bk), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND((Am != COm), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND((Bn != COn), CL_INVALID_DBK_ATTRIBUTE);

  // Check batch dimensions match.
  size_t BatchSize = TenA->rank == 3 ? TenA->shape[0] : 1;
  POCL_RETURN_ERROR_ON ((BatchSize > 1 && (BatchSize != TenB->shape[0]
                         || TenB->shape[0] != TenCOut->shape[0])),
                        CL_INVALID_DBK_SHAPE, "Batch size mismatch.\n");

  POCL_RETURN_ERROR_ON ((BatchSize > 1 && TenCIOpt
                         && TenCIOpt->shape[0] != TenCOut->shape[0]),
                        CL_INVALID_DBK_SHAPE, "Batch size mismatch.\n");

  POCL_RETURN_ERROR_ON ((TenA->dtype != TenB->dtype),
                        CL_INVALID_DBK_DATATYPE,
                        "dtype mismatch between A and B.\n");

  POCL_RETURN_ERROR_ON (TenCIOpt && (TenCIOpt->dtype != TenCOut->dtype),
                        CL_INVALID_DBK_DATATYPE,
                        "dtype mismatch between input and output C.\n");

/* the following code is LIBXSMM specific
  // TODO: We probably need to have support for mixed input/output
  // precisions to be able to fit results of large, low precision input
  // matrices. precision inputs. E.g.
  //
  //  * i8 x i8   --> i32
  //  * f16 x f16 --> f32
  POCL_RETURN_ERROR_ON ((TenA->dtype != TenCOut->dtype),
                        CL_DBK_INVALID_TYPE,
                        "Unsupported I/O dtype");
  // TODO: extend support for other data types.
  POCL_RETURN_ERROR_ON ((TenA->dtype != CL_TENSOR_FP32),
                        CL_DBK_UNAVAILABLE, "Unimplemented dtype support.\n");

  // TODO: check validity of data layouts of the tensors. Now assume
  // they are correct and they are using BLAS-like layout.
  float Alpha = 1.0f, Beta = 0.0f;
  if (AlphaAttr)
    memcpy(&Alpha, AlphaAttr, sizeof(float));
  if (BetaAttr)
    memcpy(&Beta, BetaAttr, sizeof(float));

  // libxsmm does not support arbitrary alpha and beta (for now).
  // [https://github.com/libxsmm/libxsmm/wiki/Development#longer-term-issues].
  POCL_RETURN_ERROR_ON ((Alpha != 1.0f || !(Beta == 0.0f || Beta == 1.0f)),
                        CL_DBK_UNAVAILABLE,
                        "UNIMPLEMENTED: arbitrary alpha and beta attributes");
*/

/*
  // Attributes seems to be correct - proceed to create a matmul/gemm
  // implementation.
  Kernel = (_cl_kernel *)std::calloc(1, sizeof(_cl_kernel));
  if (!Kernel)
    return LogError(CL_OUT_OF_HOST_MEMORY,
                    "Couldn't allocate storage for cl_kernel!");
  POCL_INIT_OBJECT(Kernel);
  Kernel->meta = getKernelMetadata(Program, CIOpt ? "khr_gemm" : "khr_matmul");
  Kernel->data = (void **)calloc(Program->num_devices, sizeof(void *));
  // TODO: Emit unique name for each unique DBK instance as debugging aid.
  // TODO: Does .name claim ownership?
  Kernel->name = "a_pocl_gemm_impl";
  Kernel->context = Program->context;
  Kernel->program = Program;

  assert(Kernel->meta->num_args == (3 + !!CIOpt));
  auto ArgSpace = static_cast<pocl_argument *>(
      calloc(Kernel->meta->num_args, sizeof(pocl_argument)));
  Kernel->dyn_arguments = ArgSpace;
*/

  size_t Lda = TenA.getBlasStrideInElts(0);
  size_t Ldb = TenB.getBlasStrideInElts(0);
  size_t Ldc = CO.getBlasStrideInElts(0);
  size_t ABatchStrideInElts = TenA.getBlasStrideInElts(1);
  size_t BBatchStrideInElts = TenB.getBlasStrideInElts(1);
  size_t CBatchStrideInElts = CO.getBlasStrideInElts(1);

#ifdef HAVE_LIBXSMM
  // libxsmm expects data in column-major format but we can feed it
  // row-major data by transposing the inputs and and the output.
  bool LibTransposeA = TransposeA ^ A.isBlasRowMajor();
  bool LibTransposeB = TransposeB ^ B.isBlasRowMajor();
  int Flags = (LibTransposeA ? LIBXSMM_GEMM_FLAG_TRANS_A : 0) |
              (LibTransposeB ? LIBXSMM_GEMM_FLAG_TRANS_B : 0);

  std::function<void(float *Dst, const _cl_command_node &Cmd, size_t BatchNum)>
      LoadCBatch;
  std::function<void(float *Dst, const _cl_command_node &Cmd, size_t BatchNum)>
      StoreCBatch;

  if (CIOpt && Beta != 0.0f) {
    if (CIOpt->isBlasRowMajor()) {
      // Need to convert C input to column-major.
      LoadCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                       size_t BatchNum) -> void {
        auto *CIData = getBufferDataAs<float *>(Cmd, 2);
        auto *Src = &CIData[BatchNum * CBatchStrideInElts];
        libxsmm_otrans(Batch, Src, sizeof(float), COm, COn, Ldc, COm);
      };
    } else {
      LoadCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                       size_t BatchNum) -> void {
        auto *CIData = getBufferDataAs<float *>(Cmd, 2);
        auto *Src = &CIData[BatchNum * CBatchStrideInElts];
        libxsmm_matcopy(Batch, Src, sizeof(float), COm, COn, Ldc, COm);
      };
    }
  } else {
    LoadCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                     size_t BatchNum) -> void {
      // Zero-initialize.
      libxsmm_matcopy(Batch, nullptr, sizeof(float), COm, COn, Ldc, COm);
    };
  }

  unsigned COKernelArgIdx = 2 + !!CIOpt;
  if (CO.isBlasRowMajor()) {
    // Results are always in column-major.
    StoreCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                      size_t BatchNum) -> void {
      auto *COData = getBufferDataAs<float *>(Cmd, COKernelArgIdx);
      auto *Dst = &COData[BatchNum * CBatchStrideInElts];
      libxsmm_otrans(Dst, Batch, sizeof(float), COm, COn, COm, Ldc);
    };
  } else {
    StoreCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                      size_t BatchNum) -> void {
      auto *COData = getBufferDataAs<float *>(Cmd, COKernelArgIdx);
      auto *Dst = &COData[BatchNum * CBatchStrideInElts];
      libxsmm_matcopy(Dst, Batch, sizeof(float), COm, COn, COm, Ldc);
    };
  }

  if (auto MatmulInstance = libxsmm_mmfunction<float>(Flags, COm, COn, Ak, Lda,
                                                      Ldb, COm, Alpha, Beta)) {
    auto *RunnerData = new std::function<void(_cl_command_node &)>(
        [=](_cl_command_node &Cmd) -> void {
          auto *AData = getBufferDataAs<float *>(Cmd, 0);
          auto *BData = getBufferDataAs<float *>(Cmd, 1);

          // TODO: Optimization: There is codegen for batched matmul
          //       in libxsmm we could use.
          std::vector<float> CTemp(COm * COn, 0.0f);
          for (size_t Batch = 0; Batch < BatchSize; Batch++) {
            LoadCBatch(CTemp.data(), Cmd, Batch);
            MatmulInstance(&AData[Batch * ABatchStrideInElts],
                           &BData[Batch * BBatchStrideInElts], CTemp.data());
            StoreCBatch(CTemp.data(), Cmd, Batch);
          }
        });

    Kernel->custom_runner = runDBK;
    Kernel->custom_runner_data = static_cast<void *>(RunnerData);
    Kernel->release_custom_runner_data = releaseDBK;
    pocl_program_insert_kernel_thsafe(Program, Kernel);

    return CL_SUCCESS;
  }
#endif // HAVE_LIBXSMM

  free(Kernel);
  free(ArgSpace);
  return LogError(CL_INVALID_DBK_ID, "Unsupported matmul/gemm configuration.");
}


CL_API_ENTRY char* CL_API_CALL
POname(clConfigureBuiltinKernelWithAttributesEXP)(
    cl_program Program, const char *KernelName, const void *KernelAttributes,
    cl_int *ErrCodeRet) CL_API_SUFFIX__VERSION_3_0 {

  char *InstanceName = NULL;
  cl_int errcode = CL_SUCCESS;

  POCL_GOTO_ERROR_COND((KernelName == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((!IS_CL_OBJECT_VALID(Program)), CL_INVALID_PROGRAM);

  assert(Program->num_devices != 0);
  assert(Program->num_builtin_kernels > 0);
  assert(Program->concated_builtin_names);

  POCL_GOTO_ERROR_ON((Program->build_status != CL_BUILD_NONE),
                     CL_INVALID_PROGRAM_EXECUTABLE,
                     "This must be called before clBuildProgram!");

  int FoundIdx = -1;
  for (size_t i = 0; i < Program->num_builtin_kernels; ++i) {
    if (strcmp(Program->builtin_kernel_names[i], KernelName) == 0) {
      FoundIdx = i; break;
    }
  }
  POCL_GOTO_ERROR_ON ((FoundIdx < 0), CL_INVALID_KERNEL_NAME,
                      "Can't find kernel named %s\n", KernelName);

  if (strcmp(KernelName, "khr_gemm") == 0) {
    const cl_dbk_attributes_khr_gemm *Attrs = (const cl_dbk_attributes_khr_gemm *)KernelAttributes;

    POCL_GOTO_ERROR_COND (Attrs->alpha == NULL, CL_INVALID_DBK_ATTRIBUTE);
    POCL_GOTO_ERROR_COND (Attrs->beta == NULL, CL_INVALID_DBK_ATTRIBUTE);

    // TODO: check alpha and beta values are sensible (e.g. not NaNs
    //       or infinities).

//    POCL_GOTO_ERROR_COND (isnan(Attrs->alpha), CL_DBK_INVALID_ATTRIBUTE);
//    POCL_GOTO_ERROR_COND (isnan(Attrs->beta), CL_DBK_INVALID_ATTRIBUTE);

    errcode = createGemmKernel(Program, Attrs->trans_a, Attrs->trans_b,
                               Attrs->a, Attrs->b, Attrs->c_in, Attrs->c_out,
                               Attrs->alpha, Attrs->beta, InstanceName);
  }

  if (strcmp(KernelName, "khr_matmul") == 0) {
    const cl_dbk_attributes_khr_matmul
        *Attrs = (const cl_dbk_attributes_khr_matmul *)KernelAttributes;

    errcode = createGemmKernel(Program, Attrs->trans_a, Attrs->trans_b,
                               Attrs->a, Attrs->b, NULL, Attrs->c,
                               NULL, NULL, InstanceName);

  }

ERROR:
  if (ErrCodeRet)
    *ErrCodeRet = errcode;

  return InstanceName;
}
POsym(clConfigureBuiltinKernelWithAttributesEXP)
