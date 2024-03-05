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

#ifndef POCL_SYNAPSE_KERNEL_H
#define POCL_SYNAPSE_KERNEL_H

#include <synapse_api.h>

#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <string>

struct BIKD;
struct pocl_argument_info;
struct pocl_argument;
struct _cl_command_node;

struct SynapseKernel {
  SynapseKernel(synDeviceId D, synDeviceType DT, BIKD *BIKernel)
    : Dev(D), DevType(DT), BIKernel(BIKernel) {}
  ~SynapseKernel();
  bool init();
  bool launch(const synStreamHandle Stream, _cl_command_node *Node);

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

  BIKD *BIKernel = nullptr;

  bool convertArgs();
  bool convertTensorArg(struct pocl_argument_info &Arg, unsigned ArgN);
  bool convertPODArg(struct pocl_argument_info &Arg, unsigned ArgN);
  bool setupActualArgs(_cl_command_node *Node);
};

#endif

