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

#include <algorithm>
#include <atomic>

TensorBIKD::TensorBIKD(BuiltinKernelId KernelId,
                       const char *KernelName,
                       const char *GUID,
                       const std::vector<TensorBIArg> &ArgInfos)
  : BIKD(KernelId, KernelName, {}, 0) {

  num_args = ArgInfos.size();
  arg_info = new pocl_argument_info[num_args];
  data = NULL;
  num_locals = 0;

  unsigned i = 0;
  for (auto &ArgInfo : ArgInfos) {
    arg_info[i] = ArgInfo;
    arg_info[i].name = strdup(ArgInfo.name);
    arg_info[i].type_name = strdup(ArgInfo.type_name);
    arg_info[i].access_qualifier = ArgInfo.access_qualifier;
    arg_info[i].address_qualifier = ArgInfo.address_qualifier;
    arg_info[i].type = ArgInfo.type;
    arg_info[i].type_qualifier = ArgInfo.type_qualifier;
    arg_info[i].type_size = ArgInfo.type_size;
    ++i;
  }
}


// list of Tensor Builtin Kernels
TensorBIKD PoCL_Tensor_BIDescriptors[PoCL_Tensor_BIDescriptorNum] =
{
  TensorBIKD(
  POCL_CDBI_ADD_I32,
  "pocl.add.i32", // pocl kernel name
  "add_fwd_f32", // Synapse GUID
  {TensorBIArg("int*", "input1", nullptr, {1, 1024, 1024, 1}, true, false),
   TensorBIArg("int*", "input2", nullptr, {1, 1024, 1024, 1}, true, false),
   TensorBIArg("int*", "output", nullptr, {1, 1024, 1024, 1}, false, true)}
  )
};

static std::string generateUniqueString(const std::string &input)
{
  static std::atomic<unsigned long> serialNumber{1};

  return (input + "_" + std::to_string(serialNumber++));
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

  // handle zero layouts.
  // TODO layouts might not be required or supported anymore
  const char** InputLayoutsData = InputLayouts.data();
  if (InputLayouts.size() == 0 ||
      std::all_of(InputLayouts.begin(), InputLayouts.end(),
                  [](const char* A) { return A == nullptr; }) == true)
  {
    InputLayoutsData = nullptr;
  }
  const char** OutputLayoutsData = OutputLayouts.data();
  if (OutputLayouts.size() == 0 ||
      std::all_of(OutputLayouts.begin(), OutputLayouts.end(),
                  [](const char* A) { return A == nullptr; }) == true)
  {
    OutputLayoutsData = nullptr;
  }

  // handle zero scalars
  void* ScalarArgsData = ScalarArgs.data();
  if (ScalarArgs.size() == 0)
    ScalarArgsData = nullptr;

  // create graph node
  // TODO figure out if ScalarArgs is only used as pointer at this time,
  // or if the pointed-to scalar data is actually read at this point.
  // If it's the latter, this code is broken.
  err = synNodeCreate(Graph,
                      InputTensors.data(),
                      OutputTensors.data(),
                      InputTensors.size(),
                      OutputTensors.size(),
                      ScalarArgsData,
                      ScalarArgs.size(),
                      GUID.c_str(),
                      Name.c_str(),
                      InputLayoutsData,
                      OutputLayoutsData);

  // compile the graph into recipe
  char BuildLogStorage[POCL_MAX_PATHNAME_LENGTH];
  char *BuildLogTmp = nullptr;
  if (pocl_mk_tempname(BuildLogStorage, "/tmp", ".log", NULL) == 0) {
    BuildLogTmp = BuildLogStorage;
  }
  std::string UniqueRecipeName = generateUniqueString(BIKernel->name);
  err = synGraphCompile(&Recipe, Graph, UniqueRecipeName.c_str(), BuildLogTmp);
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

  for (unsigned ArgI = 0; ArgI < Meta->num_args; ++ArgI) {
    Arg = &(Node->command.run.arguments[ArgI]);
    if (ARG_IS_LOCAL(Meta->arg_info[ArgI]))
      // No kernels with local args at the moment, should not end up here
      POCL_ABORT_UNIMPLEMENTED("Synapse: local arguments");

    else if (Meta->arg_info[ArgI].type == POCL_ARG_TYPE_POINTER) {
      // In OpenCL, it's legal to pass a NULL pointer to clSetKernelArguments.
      // In that case we must pass the same NULL forward to the kernel.
      if (Arg->value == NULL) {
        POCL_ABORT_UNIMPLEMENTED("Synapse: pointer Arg == NULL");
      } else {
        // doesn't support SVM pointers
        assert(Arg->is_svm == 0);
        cl_mem M = (*(cl_mem *)(Arg->value));
        pocl_mem_identifier P = M->device_ptrs[Node->device->global_mem_id];
        uint64_t DevPtr = (uint64_t)P.mem_ptr + Arg->offset;
        launchTensorInfoMap[ArgI]->pTensorAddress = DevPtr;
      }

    } else if (Meta->arg_info[ArgI].type == POCL_ARG_TYPE_IMAGE) {
      POCL_ABORT_UNIMPLEMENTED("almaif: image arguments");

    } else if (Meta->arg_info[ArgI].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_ABORT_UNIMPLEMENTED("almaif: sampler arguments");

    } else { // POD
      assert(Meta->arg_info[ArgI].type == POCL_ARG_TYPE_NONE);
      void* Dst = ScalarArgs.data() + ScalarArgOffsets[ArgI];
      uint64_t Size = ScalarArgSizes[ArgI];
      assert (Arg->size <= Size);
      memcpy(Dst, Arg->value, std::min(Size, Arg->size));
    }
  }

  return true;
}


bool SynapseKernel::convertTensorArg(const TensorBIArg &Arg, unsigned ArgN) {

  // name
  std::string TempName(Arg.name);
  // KernelName _ ArgName _ SequentialNumber
  std::string UniqueTensorName = generateUniqueString(Name + "_" + TempName);
  TensorNameMap[ArgN] = UniqueTensorName;

  synTensorType Type = Arg.TensorType;
  synDataType DataType = Arg.TensorDataType;
  synTensorGeometry MaxGeom = Arg.MaxGeometry;

  // create Tensor
  synStatus err = synSuccess;
  synTensor TempTensor = nullptr;
  err = synTensorHandleCreate(&TempTensor, Graph,
                              Type, UniqueTensorName.c_str());
  if (err != synSuccess) {
    POCL_MSG_ERR("SynapseKernel: failed to create Tensor for Arg %u: %s\n",
                 ArgN, Arg.name);
    return false;
  }

  err = synTensorSetDeviceDataType(TempTensor, DataType);
  err = synTensorSetGeometry(TempTensor, &MaxGeom, synGeometryMaxSizes);

  // store Tensor
  TensorMap[ArgN] = TempTensor;
  // TODO can tensors be both input/output ?
  if (Arg.Input) {
    InputTensors.push_back(TempTensor);
    InputLayouts.push_back(Arg.DataLayout);
  } else {
    assert(Arg.Output == true);
    OutputTensors.push_back(TempTensor);
    OutputLayouts.push_back(Arg.DataLayout);
  }

  // fill out launch info as much as possible
  synLaunchTensorInfo TempLaunchInfo;
  std::memset(&TempLaunchInfo, 0, sizeof(TempLaunchInfo));
  TempLaunchInfo.tensorName     = TensorNameMap[ArgN].c_str();
  TempLaunchInfo.tensorType     = Type;
  launchTensorInfo.push_back(TempLaunchInfo);
  auto It = std::prev(launchTensorInfo.end());
  launchTensorInfoMap[ArgN] = &*It;

  return true;
}

bool SynapseKernel::convertPODArg(const TensorBIArg &Arg, unsigned ArgN) {
  if (Arg.type_size == 0) {
    POCL_MSG_ERR("Synapse: Type size for Arg %u is zero\n", ArgN);
    return false;
  }
  uint64_t Offset = ScalarArgs.size();
  uint64_t Alignment = pocl_size_ceil2_64(Arg.type_size);
  Offset = align64(Offset, Alignment);
  uint64_t TotalSize = Offset + Arg.type_size;
  ScalarArgs.resize(TotalSize);
  ScalarArgOffsets[ArgN] = Offset;
  ScalarArgSizes[ArgN] = Arg.type_size;
  return true;
}

bool SynapseKernel::convertArgs() {
  synStatus err = synSuccess;

  for (unsigned i = 0; i < BIKernel->num_args ; ++i) {
    // TODO get the type from BIKD
    const TensorBIArg &Arg = BIKernel->TensorArgInfos[i];
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
                           synDeviceId DevID,
                           cl_kernel Kernel,
                           struct pocl_context *PoclContext,
                           struct _cl_command_node *Node) {

  struct pocl_argument *Arg;

  if (PoclContext->num_groups[0] == 0
      || PoclContext->num_groups[1] == 0
      || PoclContext->num_groups[2] == 0)
    return true;

  setupActualArgs(Node);


  //Schedule kernel
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

  err = synDeviceSynchronize(DevID);
  if (err != synSuccess) {
    POCL_MSG_ERR("SynapseKernel: failed to sync after recipe launch\n");
    return false;
  }

  POCL_MEM_FREE(Node->command.run.device_data);

  return true;
}
