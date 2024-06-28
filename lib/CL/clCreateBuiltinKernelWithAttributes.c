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


