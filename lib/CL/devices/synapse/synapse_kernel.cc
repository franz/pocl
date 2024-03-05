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

#include "synapse_kernel.hh"

#include "pocl_cl.h"

#include "builtin_kernels.hh"

#include "pocl_util.h"
#include "pocl_file_util.h"

/*
struct SynapseKernel {
  SynapseKernel(synDeviceId D, synDeviceType DT, BIKD *BIKernel)
    : Dev(D), DevType(DT), Kernel(BIKernel) {}
  ~SynapseKernel();
  bool init();
  bool launch(const synStreamHandle Stream);

private:
  synDeviceType DevType = synDeviceTypeInvalid;
  synDeviceId Dev = (synDeviceId)(-1);

  synGraphHandle Graph = nullptr;
  // compiled graph
  synRecipeHandle Recipe = nullptr;
  // Workspace Memory
  uint64_t WorkspaceMem = 0;
  uint64_t WorkspaceSize = 0;

  std::map<std::string, synTensor> TensorMap;
  std::vector<synTensor> InputTensors;
  std::vector<synTensor> OutputTensors;
  std::vector<uint8_t> ScalarArgs;
  std::map<unsigned, size_t> ScalarArgOffsets;
  std::map<unsigned, size_t> ScalarArgSizes;
  std::vector<const char*> InputLayouts;
  std::vector<const char*> OutputLayouts;
  std::string BuildLog;
  std::string Name;
  std::string GUID;
  std::vector<synLaunchTensorInfo> launchTensorInfo;

  BIKD *Kernel = nullptr;

  bool convertArgs();
  bool convertTensorArg(struct pocl_argument_info &Arg, unsigned ArgN);
  bool convertPODArg(struct pocl_argument_info &Arg, unsigned ArgN);
  bool setupActualArgs(struct pocl_argument *Args);
};

*/

static synDataType getSynArgType(const char* Typename) {
  std::string ArgType(Typename);
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
  // invalid type
  return syn_type_na;
}

static inline uint64_t
align64(uint64_t value, unsigned alignment)
{
  return (value + alignment - 1) & ~((uint64_t)alignment - 1);
}

bool SynapseKernel::init() {
  synStatus err = synGraphCreate(&Graph, DevType);
  if (err != synSuccess) {
    POCL_MSG_ERR("SynapseKernel: failed to create graph for device %u",
                 (unsigned)DevType);
    return false;
  }

  Name = BIKernel->name;
  // GUID is the "operator" name. Could be a builtin operator, or if not found
  // in the list of builtin operators, will be forwarded to plugin
  GUID = BIKernel->GUID;
  // convert OpenCL arguments to Synapse arguments
  if (!convertArgs())
    return false;

  // create graph node
  // TODO figure out if ScalarArgs is only used as pointer at this time,
  // or if the pointed-to scalar data is actually read at this point.
  // If it's the latter, this code is broken.
  err = synNodeCreate(Graph,
                      InputTensors.data(),
                      OutputTensors.data(),
                      InputTensors.size(),
                      OutputTensors.size(),
                      ScalarArgs.data(),
                      ScalarArgs.size(),
                      GUID.c_str(),
                      Name.c_str(),
                      InputLayouts.data(),
                      OutputLayouts.data());

  // compile the graph into recipe
  char BuildLogStorage[POCL_MAX_PATHNAME_LENGTH];
  char *BuildLogTmp = nullptr;
  if (pocl_mk_tempname(BuildLogStorage, "/tmp", ".log", NULL) == 0)
    BuildLogTmp = BuildLogStorage;
  // TODO unique recipe name!
  err = synGraphCompile(&Recipe, Graph, BIKernel->name, BuildLogTmp);
  if (BuildLogTmp) {
    char *Content = nullptr;
    uint64_t Size = 0;
    if (pocl_read_file(BuildLogTmp, &Content, &Size) == 0 && (Size > 0)) {
      BuildLog = std::string(Content, Size);
      free(Content);
    }
  }
  if (err != synSuccess) {
    Recipe = nullptr;
    POCL_MSG_ERR("SynapseKernel: failed to compile Graph for Kernel %s\n",
                 BIKernel->name);
    POCL_MSG_ERR("SynapseKernel: compilation build log: %s\n",
                 BuildLog.c_str());
    return false;
  }

  //create Workspace for all the non-user-managed memory
  err = synWorkspaceGetSize(&WorkspaceSize, Recipe);
  if (err != synSuccess) {
      POCL_MSG_ERR("SynapseKernel: failed to query workspace size\n");
      return false;
  }
  if (WorkspaceSize > 0)
  {
      err = synDeviceMalloc(Dev, WorkspaceSize, 0, 0, &WorkspaceMem);
      if (err != synSuccess) {
          POCL_MSG_ERR("SynapseKernel: failed to allocate workspace mem\n");
          return false;
      }
  }

  return true;
}

bool SynapseKernel::setupActualArgs(struct _cl_command_node *Node
                                    ) {

  struct pocl_argument *Arg;
  cl_kernel Kernel = Node->command.run.kernel;
  pocl_kernel_metadata_t *Meta = Kernel->meta;
  struct pocl_context *PoclContext = &Node->command.run.pc;

  unsigned CurrentScalarArg = 0;
  unsigned CurrentPointerArg = 0;

  for (unsigned ArgI = 0; ArgI < Meta->num_args; ++ArgI) {
    Arg = &(Node->command.run.arguments[ArgI]);
    if (ARG_IS_LOCAL(Meta->arg_info[ArgI]))
      // No kernels with local args at the moment, should not end up here
      POCL_ABORT_UNIMPLEMENTED("Synapse: local arguments");

    else if (Meta->arg_info[ArgI].type == POCL_ARG_TYPE_POINTER) {
      // It's legal to pass a NULL pointer to clSetKernelArguments. In
      //   that case we must pass the same NULL forward to the kernel.
      if (Arg->value == NULL) {
        POCL_ABORT_UNIMPLEMENTED("Synapse: pointer Args == NULL");
      } else {
        // doesn't support SVM pointers
        assert(Arg->is_svm == 0);
        cl_mem M = (*(cl_mem *)(Arg->value));
        pocl_mem_identifier P = M->device_ptrs[Node->device->global_mem_id];
        uint64_t DevPtr = (uint64_t)P.mem_ptr + Arg->offset;
        // TODO
        *(size_t *)current_arg = buffer;
      }

    } else if (Meta->arg_info[ArgI].type == POCL_ARG_TYPE_IMAGE) {
      POCL_ABORT_UNIMPLEMENTED("almaif: image arguments");

    } else if (Meta->arg_info[ArgI].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_ABORT_UNIMPLEMENTED("almaif: sampler arguments");

    } else { // POD
      void* Dst = ScalarArgs.data() + ScalarArgOffsets[ArgI];
      uint64_t Size = ScalarArgSizes[ArgI];
      assert (Arg->size <= Size);
      memcpy(Dst, Arg->value, std::min(Size, Arg->size));
    }
  }

  synStatus err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);

  POCL_MEM_FREE(Node->command.run.device_data);

  return true;
}


bool SynapseKernel::convertTensorArg(pocl_argument_info &Arg, unsigned ArgN) {
  synTensorDescriptor Desc;
  std::memset(&Desc, 0, sizeof(synTensorDescriptor));

  // name
  std::string ArgName(Arg.name);
  Desc.m_name = Arg.name;

  // data type
  std::string ArgType(Arg.type_name);
  Desc.m_dataType = getSynArgType(Arg.type_name);
  if (Desc.m_dataType == syn_type_na) {
    POCL_MSG_ERR("SynapseKernel: unrecognized data type for Arg %u: %s\n",
                 ArgN, Arg.type_name);
    return false;
  }

  // TODO we need to somehow get these from BIKD
  // Tensor dimensions, for example:
  //    unsigned inTensorSize[SYN_MAX_TENSOR_DIM]  = {inZ, inW, inH, batch};
  //    unsigned outTensorSize[SYN_MAX_TENSOR_DIM] = {inZ, outW, outH, batch};
  // Tensor is input/output arg

  // dimensions & sizes for each dim
  synTensorGeometry Geom;
  Desc.m_dims = 1;
  Geom.dims = Desc.m_dims;
  for (unsigned j = 0; j < Desc.m_dims; ++j)
  {
      Desc.m_sizes[j]    =  0; // inTensorSize[j];
      Desc.m_minSizes[j] = Desc.m_sizes[j];
      Geom.sizes[j] = Desc.m_sizes[j];
  }

  // derive input or output from presence of "const"
  bool TensorIsInput = (Arg.type_qualifier & CL_KERNEL_ARG_TYPE_CONST);

  if (TensorIsInput) {
    Desc.m_isInput = true;
    Desc.m_isOutput = false;
  } else {
    Desc.m_isInput = false;
    Desc.m_isOutput = true;
  }

  // create Tensor
  synStatus err = synSuccess;
  synTensor TempTensor = nullptr;
  err = synTensorHandleCreate(&TempTensor, Graph, DATA_TENSOR, TensorName);
  if (err != synSuccess) {
    POCL_MSG_ERR("SynapseKernel: failed to create Tensor for Arg %u: %s\n",
                 ArgN, Arg.name);
    return false;
  }

  err = synTensorSetDeviceDataType(TempTensor, Desc.m_dataType);
  err = synTensorSetGeometry(TempTensor, &Geom, synGeometryMaxSizes);

  // store Tensor
  TensorMap[ArgN] = TempTensor;
  if (TensorIsInput) {
    InputTensors.push_back(TempTensor);
  } else {
    OutputTensors.push_back(TempTensor);
  }
  synLaunchTensorInfo TempLaunchInfo;
//  persistentTensorInfo[0].pTensorAddress = pDeviceInputA;
//  persistentTensorInfo[0].tensorName     = "inputA";
//  persistentTensorInfo[0].tensorType     = DATA_TENSOR;
//  persistentTensorInfo[0].tensorId       = 0;

  std::memset(&TempLaunchInfo, 0, sizeof(TempLaunchInfo));
  TempLaunchInfo.tensorName = TensorName;
  TempLaunchInfo.tensorType     = DATA_TENSOR;
  launchTensorInfo.push_back(TempLaunchInfo);

  return true;
}

bool SynapseKernel::convertPODArg(struct pocl_argument_info &Arg, unsigned ArgN) {
  if (Arg.type_size == 0) {
    POCL_MSG_ERR("Synapse: Type size for Arg %u is zero\n", ArgN);
    return false;
  }
  uint64_t Offset = ScalarArgs.size();
  uint64_t Alignment = pocl_size_ceil2_64(Arg.type_size);
  Offset = align64(Offset, Alignment);
  ScalarArgOffsets[ArgN] = Offset;
  ScalarArgSizes[ArgN] = Arg.type_size;
  return true;
}

bool SynapseKernel::convertArgs() {
  synStatus err = synSuccess;

  for (unsigned i = 0; i < BIKernel->num_args ; ++i) {
    // TODO get the type from BIKD
    struct pocl_argument_info &Arg = BIKernel->arg_info[i];
    if (Arg.type == POCL_ARG_TYPE_NONE) {
      if (!convertPODArg(Arg, i))
        return false;
    } else if (Arg.type == POCL_ARG_TYPE_POINTER) {
      if (!convertTensorArg(Arg, i))
        return false;
    } else {
      POCL_MSG_ERR("SynapseKernel: Can't handle types other "
                   "than POD & Pointers\n");
      return false;
    }
  }
  return true;
}

SynapseKernel::~SynapseKernel() {
  synStatus err = synSuccess;
  if (WorkspaceMem) {
    err = synDeviceFree(Dev, WorkspaceMem, 0);
    if (err != synSuccess) POCL_MSG_ERR("Failed to release workspace mem\n");
  }
  if (Recipe) {
    err = synRecipeDestroy(Recipe);
    if (err != synSuccess) POCL_MSG_ERR("Failed to release recipe\n");
  }
  if (Graph) {
    err = synGraphDestroy(Graph);
    if (err != synSuccess) POCL_MSG_ERR("Failed to release graph\n");
  }
  // according to Docs, synGraphDestroy should destroy all Tensors as well
//  for (auto Iter : TensorMap) {
//    err = synTensorDestroy(Iter.second);
//    if (err != synSuccess) POCL_MSG_ERR("Failed to release graph\n");
//  }
  TensorMap.clear();
  InputTensors.clear();
  OutputTensors.clear();
  InputLayouts.clear();
  OutputLayouts.clear();
}

bool SynapseKernel::launch(const synStreamHandle Stream,
                           struct _cl_command_node *Node) {

  struct pocl_argument *Arg;
  cl_kernel kernel = Node->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *PoclContext = &Node->command.run.pc;

  if (PoclContext->num_groups[0] == 0
      || PoclContext->num_groups[1] == 0
      || PoclContext->num_groups[2] == 0)
    return true;

  setupActualArgs(Node);
  //Associate the tensors with the device memory so compute knows where to read from / write to
  //Schedule kernel
  std::vector<synLaunchTensorInfo> launchTensorInfo;

  synStatus err = synLaunch(Stream,
                            launchTensorInfo.data(),
                            launchTensorInfo.size(),
                            WorkspaceMem,
                            Recipe,
                            0); // flags
  // possible flags: SYN_FLAGS_TENSOR_NAME: identify the tensors by their names
  if (err != synSuccess) {
    POCL_MSG_ERR("SynapseKernel: failed to launch recipe\n");
    return false;
  }


  return true;
}
