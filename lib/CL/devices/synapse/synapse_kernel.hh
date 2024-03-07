/* SynapseKernel - implements builtin kernels through Synapse API

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

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

#include "builtin_kernels.hh"

#include <synapse_api.h>

#ifndef POCL_SYNAPSE_KERNEL_H
#define POCL_SYNAPSE_KERNEL_H

#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <string>

// An initialization wrapper for kernel argument metadata;
// this class supports both Tensor and other (Scalar) arguments
struct TensorBIArg : public BIArg
{
  const char* DataLayout;
  synTensorGeometry MaxGeometry;
  synTensorType TensorType;
  synDataType TensorDataType;
  bool Input, Output, isTensor;

  // constructor for Tensor args
  TensorBIArg (const char *TypeName, const char *Name,
               const char *Layout, std::vector<unsigned> Geometry, // geometry = Z, W, H, batch
               bool isInput, bool isOutput, synTensorType TensorT = DATA_TENSOR
               ) : BIArg(TypeName, Name, POCL_ARG_TYPE_POINTER,
                         CL_KERNEL_ARG_ADDRESS_GLOBAL,
                         CL_KERNEL_ARG_ACCESS_NONE,
                         CL_KERNEL_ARG_TYPE_NONE,
                         0) // size
  {
    isTensor = true;
    DataLayout = Layout;
    MaxGeometry.dims = Geometry.size();
    for (unsigned i = 0; i < Geometry.size(); ++i)
      MaxGeometry.sizes[i] = Geometry[i];
    Input = isInput;
    Output = isOutput;
    TensorType = TensorT;
    TensorDataType = getTensorDataType();
  }

  // constructor for Scalar args
  TensorBIArg (const char *TypeName, const char *Name, pocl_argument_type Type,
         cl_kernel_arg_address_qualifier ADQ = CL_KERNEL_ARG_ADDRESS_GLOBAL,
         cl_kernel_arg_access_qualifier ACQ = CL_KERNEL_ARG_ACCESS_NONE,
         cl_kernel_arg_type_qualifier TQ = CL_KERNEL_ARG_TYPE_NONE,
         size_t size = 0) : BIArg(TypeName, Name, Type, ADQ, ACQ, TQ, size) {
    isTensor = false;
    TensorDataType = syn_type_na;
    TensorType = TENSOR_TYPE_INVALID;
    MaxGeometry.dims = 0;
    DataLayout = nullptr;
  }

  ~TensorBIArg () {}

private:
  synDataType getTensorDataType() const {
    std::string ArgType(type_name);
    if (ArgType == "char") {
      return syn_type_int8;
    }
    if (ArgType == "uchar") {
      return syn_type_uint8;
    }
    if (ArgType == "short") {
      return syn_type_int16;
    }
    if (ArgType == "ushort") {
      return syn_type_uint16;
    }
    if (ArgType == "int32") {
      return syn_type_int32;
    }
    if (ArgType == "uint32") {
      return syn_type_uint32;
    }
    if (ArgType == "int64") {
      return syn_type_int64;
    }
    if (ArgType == "uint64") {
      return syn_type_uint64;
    }
    if (ArgType == "float") {
      return syn_type_float;
    }
    if (ArgType == "half") {
      return syn_type_fp16;
    }
    // invalid or unrecognized type
    return syn_type_na;
  }
};

// An initialization wrapper for kernel metadatas.
// BIKD = Built-in Kernel Descriptor
struct TensorBIKD : public BIKD
{
  const char* GUID;
  const std::vector<TensorBIArg> TensorArgInfos;

  TensorBIKD(BuiltinKernelId KernelId,
             const char *KernelName,
             const char *GUID,
             const std::vector<TensorBIArg> &ArgInfos);

  ~TensorBIKD() {
    for (size_t i = 0; i < num_args; ++i) {
      free(arg_info[i].name);
      free(arg_info[i].type_name);
    }
    delete[] arg_info;
    free (name);
  }

  BuiltinKernelId KernelId;
};

constexpr unsigned PoCL_Tensor_BIDescriptorNum = 1;
extern TensorBIKD PoCL_Tensor_BIDescriptors[PoCL_Tensor_BIDescriptorNum];


struct SynapseKernel {
  SynapseKernel(synDeviceId D, synDeviceType DT, TensorBIKD *BIK)
    : Dev(D), DevType(DT), BIKernel(BIK) {}
  ~SynapseKernel();
  bool init();
  bool launch(const synStreamHandle Stream, synDeviceId DevID,
              cl_kernel Kernel,
              struct pocl_context *PoclContext,
              _cl_command_node *Node);

private:
  synDeviceType DevType = synDeviceTypeInvalid;
  synDeviceId Dev = (synDeviceId)(-1);


  synGraphHandle Graph = nullptr;
  // compiled graph
  synRecipeHandle Recipe = nullptr;
  // Workspace Memory
  uint64_t WorkspaceMem = 0;
  uint64_t WorkspaceSize = 0;

  //this holds the association of the tensors with the device memory
  std::vector<synLaunchTensorInfo> launchTensorInfo;
  // ArgN -> pointer into ^^
  std::map<unsigned, synLaunchTensorInfo*> launchTensorInfoMap;

  // ArgN -> tensors
  std::map<unsigned, synTensor> TensorMap;
  std::map<unsigned, std::string> TensorNameMap;
  // tensors by direction
  std::vector<synTensor> InputTensors;
  std::vector<synTensor> OutputTensors;
  // tensor layout strings
  std::vector<const char*> InputLayouts;
  std::vector<const char*> OutputLayouts;

  // storage for scalar arguments
  std::vector<uint8_t> ScalarArgs;
  // scalar argument offsets (into ^^storage), indexed by ArgN, aligned
  std::map<unsigned, size_t> ScalarArgOffsets;
  // scalar argument sizes, indexed by ArgN
  std::map<unsigned, size_t> ScalarArgSizes;

  // compilation log
  std::string BuildLog;
  std::string Name;
  std::string GUID;

  TensorBIKD *BIKernel = nullptr;

  bool convertArgs();
  bool convertTensorArg(const TensorBIArg &Arg, unsigned ArgN);
  bool convertPODArg(const TensorBIArg &Arg, unsigned ArgN);
  bool setupActualArgs(_cl_command_node *Node);
};

#endif

