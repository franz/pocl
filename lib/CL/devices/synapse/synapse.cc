/* - driver for Synapse API (used by Gaudi accelerator)

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

#include "synapse.h"

//#include "AlmaifCompile.hh"

//#include "bufalloc.h"
#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl_cl.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "pocl_file_util.h"

#include "builtin_kernels.hh"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <climits>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>

#include <synapse_api.h>

//extern int pocl_offline_compile;

static cl_bool DevAvailable = CL_TRUE;
static cl_bool DevUNAvailable = CL_FALSE;

struct SynapseDeviceData {
  std::string Params;
  std::string SupportedList;

  std::set<BIKD *> SupportedKernels;

  // wakeup driver thread
  pocl_cond_t WakeupCond;
  pocl_lock_t WakeupLock;

  // List of commands ready to be executed.
  _cl_command_node *WorkQueue = nullptr;

  // driver thread
  pocl_thread_t DriverThread = 0;
  bool ExitRequested = false;

  //synStreamHandle Dev2HostStream = 0, Host2DevStream = 0, ComputeStream = 0;
  synStreamHandle DefaultStream = 0;
  synDeviceId DevID = 0;
};

struct SynapseCond {
  pocl_cond_t Cond;

  SynapseCond() {
    POCL_INIT_COND(Cond);
  }

  ~SynapseCond() {
    POCL_DESTROY_COND(Cond);
  }

  void wait(pocl_lock_t &Lock) {
    POCL_WAIT_COND(Cond, Lock);
  }

  void signal() {
    POCL_SIGNAL_COND(Cond);
  }

  void broadcast() {
    POCL_BROADCAST_COND(Cond);
  }
};

struct SynapseKernel {
  synDeviceType DevType = synDeviceTypeInvalid;

  synGraphHandle Graph = nullptr;
  // compiled graph
  synRecipeHandle Recipe = nullptr;
  // Workspace Memory
  uint64_t WorkspaceMem = 0;
  uint64_t WorkspaceSize = 0;

  std::map<std::string, synTensor> TensorMap;


  SynapseKernel(synDeviceType DT) : DevType(DT) {}
  ~SynapseKernel() {}
  bool init(BIKD *BIKernel);
  bool launch();
};

bool SynapseKernel::init(BIKD *BIKernel) {
  synStatus err = synGraphCreate(&Graph, DevType);
  if (err != synSuccess) {
    POCL_MSG_ERR("SynapseKernel: failed to create graph for device %u",
                 (unsigned)DevType);
    return false;
  }

  synTensorDescriptor Desc;
  std::memset(&Desc, 0, sizeof(synTensorDescriptor));
  for (unsigned i = 0; i < BIKernel->num_args ; ++i) {
    // TODO get the type from BIKD
    struct pocl_argument_info &Arg = BIKernel->arg_info[i];
    if (Arg.type != POCL_ARG_TYPE_POINTER) {
      POCL_MSG_ERR("SynapseKernel: Can't handle non-pointer types yet\n");
    }
    Desc.m_name = Arg.name;
    std::string ArgName(Arg.name);
    std::string ArgType(Arg.type_name);
    {
      if (ArgType == "char") {
        Desc.m_dataType     = syn_type_int8;
      }
      if (ArgType == "uchar") {
        Desc.m_dataType     = syn_type_uint8;
      }
      if (ArgType == "short") {
        Desc.m_dataType     = syn_type_int16;
      }
      if (ArgType == "ushort") {
        Desc.m_dataType     = syn_type_uint16;
      }
      if (ArgType == "int32") {
        Desc.m_dataType     = syn_type_int32;
      }
      if (ArgType == "uint32") {
        Desc.m_dataType     = syn_type_uint32;
      }
      if (ArgType == "int64") {
        Desc.m_dataType     = syn_type_int64;
      }
      if (ArgType == "uint64") {
        Desc.m_dataType     = syn_type_uint64;
      }
      if (ArgType == "float") {
        Desc.m_dataType     = syn_type_float;
      }
      if (ArgType == "half") {
        Desc.m_dataType     = syn_type_fp16;
      }
    }

    // TODO we need to somehow get these from  BIKD
    // examples:
    //    unsigned inTensorSize[SYN_MAX_TENSOR_DIM]  = {inZ, inW, inH, batch};
    //    unsigned outTensorSize[SYN_MAX_TENSOR_DIM] = {inZ, outW, outH, batch};

    Desc.m_dims = 1;
    for (unsigned j = 0; j < Desc.m_dims; ++j)
    {
        Desc.m_sizes[j]    =  0; // inTensorSize[j];
        Desc.m_minSizes[j] = Desc.m_sizes[j];
    }
    synTensor TempTensor = nullptr;
    err = synTensorHandleCreate(&TempTensor, Graph, DATA_TENSOR, nullptr);
    if (err != synSuccess) {
      POCL_MSG_ERR("SynapseKernel: failed to create Tensor for Argument %u %s",
                   i, Arg.name);
      return false;
    }
    TensorMap[ArgName] = TempTensor;
    TempTensor = nullptr;
  }

  err = synNodeCreate();

  // compile the graph into recipe
  char BuildLogStorage[POCL_MAX_PATHNAME_LENGTH];
  char *BuildLog = nullptr;
  if (pocl_mk_tempname(BuildLogStorage, "/tmp", ".log", NULL) == 0)
    BuildLog = BuildLogStorage;
  err = synGraphCompile(&Recipe, Graph, BIKernel->name, BuildLog);
  if (err != synSuccess) {
    Recipe = nullptr;
    POCL_MSG_ERR("SynapseKernel: failed to compile Graph for Kernel %s",
                 BIKernel->name);
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
      err = synDeviceMalloc(D->DevID, WorkspaceSize, 0, 0, &WorkspaceMem);
      if (err != synSuccess) {
          POCL_MSG_ERR("SynapseKernel: failed to allocate workspace mem\n");
          return false;
      }
  }
}

bool SynapseKernel::launch() {
  //Associate the tensors with the device memory so compute knows where to read from / write to
  synLaunchTensorInfo persistentTensorInfo[3];
  persistentTensorInfo[0].pTensorAddress = pDeviceInputA;
  persistentTensorInfo[0].tensorName     = "inputA"; //Must match the name supplied at tensor creation
  persistentTensorInfo[0].tensorType     = DATA_TENSOR;
  persistentTensorInfo[1].pTensorAddress = pDeviceInputB;
  persistentTensorInfo[1].tensorName     = "inputB"; //Must match the name supplied at tensor creation
  persistentTensorInfo[1].tensorType     = DATA_TENSOR;
  persistentTensorInfo[2].pTensorAddress = pDeviceOutput;
  persistentTensorInfo[2].tensorName     = "output"; //Must match the name supplied at tensor creation
  persistentTensorInfo[2].tensorType     = DATA_TENSOR;
  //Schedule compute
  synStatus err = synLaunch(D->DefaultStream,
                            persistentTensorInfo,
                            3,
                            WorkspaceMem,
                            Recipe);
  if (err != synSuccess) {
    POCL_MSG_ERR("SynapseKernel: failed to launch recipe\n");
    return false;
  }

}

void pocl_synapse_init_device_ops(struct pocl_device_ops *ops) {

  ops->device_name = "synapse";
  ops->init = pocl_synapse_init;
  ops->uninit = pocl_synapse_uninit;
  ops->probe = pocl_synapse_probe;
  ops->build_hash = pocl_synapse_build_hash;
  //ops->setup_metadata = pocl_setup_builtin_metadata;

  /* TODO: Bufalloc-based allocation from the onchip memories. */
  ops->alloc_mem_obj = pocl_synapse_alloc_mem_obj;
  ops->free = pocl_synapse_free;

  ops->write = pocl_synapse_write;
  ops->read = pocl_synapse_read;
  ops->copy = pocl_synapse_copy;
  ops->copy_rect = pocl_synapse_copy_rect;
  ops->read_rect = pocl_synapse_read_rect;
  ops->write_rect = pocl_synapse_write_rect;
  ops->memfill = pocl_synapse_memfill;

  ops->map_mem = pocl_synapse_map_mem;
  ops->unmap_mem = pocl_synapse_unmap_mem;
  ops->get_mapping_ptr = pocl_synapse_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_synapse_free_mapping_ptr;

  ops->submit = pocl_synapse_submit;
  ops->join = pocl_synapse_join;
  ops->notify = pocl_synapse_notify;
  ops->broadcast = pocl_broadcast;
  ops->run = nullptr;
  ops->free_event_data = pocl_synapse_free_event_data;
  ops->update_event = pocl_synapse_update_event;
  ops->wait_event = pocl_synapse_wait_event;
  ops->notify_event_finished = pocl_synapse_notify_event_finished;
  ops->notify_cmdq_finished = pocl_synapse_notify_cmdq_finished;
  ops->init_queue = pocl_synapse_init_queue;
  ops->free_queue = pocl_synapse_free_queue;

  //ops->build_builtin = pocl_driver_build_opencl_builtins;
  //ops->free_program = pocl_driver_free_program;
}

static uint32_t DeviceCountsByType[synDeviceTypeSize] = {0};
static uint32_t UsableDevs = 0;
static uint32_t AcquiredDevs = 0;

static void *synapseDriverThread(void *dev);

unsigned int pocl_synapse_probe(struct pocl_device_ops *ops) {
  int env_count = pocl_device_get_env_count(ops->device_name);

  // env var must be set & nonzero to enable the driver
  if (env_count <= 0) {
    return 0;
  }

  synStatus err;
  err = synInitialize();
  if (err != synSuccess)
  {
    POCL_MSG_ERR("Failed to initialize Synapse API\n");
    return 0;
  }

  err = synDeviceCount(DeviceCountsByType);
  if (err != synSuccess)
  {
      POCL_MSG_ERR("Failed to query devices\n");
      return 0;
  }

  UsableDevs = DeviceCountsByType[synDeviceGaudi]
               + DeviceCountsByType[synDeviceGaudi2];

  if (env_count > UsableDevs)
  {
    POCL_MSG_WARN("Env count %i is > than # available accelerators %u",
                  env_count, UsableDevs);
    env_count = UsableDevs;
  }

  POCL_MSG_PRINT_SYNAPSE("Detected %i Gaudi+Gaudi2 devices", UsableDevs);
  return env_count;
}

char *pocl_synapse_build_hash(cl_device_id device) {
  SynapseDeviceData *D = (SynapseDeviceData *)device->data;
  char *Res = (char *)calloc(1000, sizeof(char));
  snprintf(Res, 1000, "synapse-%s", D->Params.c_str());
  return Res;
}

cl_int pocl_synapse_init(unsigned j, cl_device_id dev, const char *parameters) {

  assert(j < UsableDevs);
  SETUP_DEVICE_CL_VERSION(dev, 1, 2);
  dev->type = CL_DEVICE_TYPE_CUSTOM;
  dev->long_name = (char *)"Synapse API accelerator";
  dev->short_name = "synapse";
  dev->vendor = "PoCL";
  dev->version = "1.2";
  dev->extensions = "";
  dev->profiling_timer_resolution = 1000;
  dev->profile = "EMBEDDED_PROFILE";
  dev->max_mem_alloc_size = 1UL << 32;
  // needs 128 byte alignment (= 1024 in bits)
  dev->mem_base_addr_align = 1024;
  dev->max_constant_buffer_size = 32768;
  dev->local_mem_size = 16384;
  dev->max_work_item_dimensions = 3;
  // kernel param size (bytes). TODO find from API
  dev->max_parameter_size = 128;
  dev->address_bits = 32;

  dev->max_work_item_dimensions = 3;
  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1] =
      dev->max_work_item_sizes[2] = dev->max_work_group_size = 64;
  dev->preferred_wg_size_multiple = 8;
  dev->endian_little = CL_TRUE;

  SynapseDeviceData *D = new SynapseDeviceData;
  dev->available = &DevAvailable;
  dev->data = (void *)D;

  dev->compiler_available = CL_FALSE;
  dev->linker_available = CL_FALSE;

  dev->device_side_printf = 0;
  dev->printf_buffer_size = 0;

  synStatus err = synSuccess;
  synDeviceType ToAcq = synDeviceTypeInvalid;
  if (DeviceCountsByType[synDeviceGaudi] > 0) {
    ToAcq = synDeviceGaudi;
    --DeviceCountsByType[synDeviceGaudi];
  } else if (DeviceCountsByType[synDeviceGaudi2] > 0) {
    ToAcq = synDeviceGaudi2;
    --DeviceCountsByType[synDeviceGaudi2];
  } else {
    POCL_MSG_ERR("Synapse: BUG ran out of devices to initialize\n");
    err = synDeviceTypeMismatch;
  }
  if (err == synSuccess) {
    err = synDeviceAcquireByDeviceType(&D->DevID, ToAcq);
    if (err != synSuccess) {
      POCL_MSG_ERR("Synapse: failed to acquire device by type");
      dev->available = &DevUNAvailable;
      return 0;
    }
  }

  ++AcquiredDevs;

  uint64_t FreeMem = 0, TotalMem = 0;
  err = synDeviceGetMemoryInfo(D->DevID, &FreeMem, &TotalMem);
  if (err == synSuccess) {
    dev->global_mem_size = TotalMem;
  } else {
    POCL_MSG_ERR("Synapse: failed to get device memory info");
    dev->available = &DevUNAvailable;
    return 0;
  }

/*
  err = synStreamCreateGeneric(&D->ComputeStream, D->DevID, 0);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synStreamCreateGeneric failed with %u\n",
                 (unsigned)err);
    dev->available = &DevUNAvailable;
    return 0;
  }
  err = synStreamCreateGeneric(&D->Dev2HostStream, D->DevID, 0);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synStreamCreateGeneric failed with %u\n",
                 (unsigned)err);
    dev->available = &DevUNAvailable;
    return 0;
  }
  err = synStreamCreateGeneric(&D->Host2DevStream, D->DevID, 0);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synStreamCreateGeneric failed with %u\n",
                 (unsigned)err);
    dev->available = &DevUNAvailable;
    return 0;
  }
*/
  err = synStreamCreateGeneric(&D->DefaultStream, D->DevID, 0);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synStreamCreateGeneric failed with %u\n",
                 (unsigned)err);
    dev->available = &DevUNAvailable;
    return 0;
  }

  synDeviceInfoV2 DevInfo;
  err = synDeviceGetInfoV2(D->DevID, &DevInfo);
  if (err == synSuccess) {
    dev->local_mem_size = DevInfo.sramSize;
  } else {
    POCL_MSG_ERR("Synapse: failed to get device info V2");
    dev->local_mem_size = 8192;
  }

  D->Params = parameters;
  if (!parameters) {
    POCL_MSG_PRINT_SYNAPSE("Parameters were not given. Creating emulation "
                          "device with built-in kernels 0,1,2,4,9.\n");
    D->Params = "0,1,2,4,9";
  }

  std::string IterStr(D->Params);
  while (!IterStr.empty()) {
    auto DelimIt = IterStr.find(",");
    std::string Token = IterStr.substr(0, DelimIt);
    IterStr.erase(0, DelimIt);

    int BIKcode = std::stoi(Token);
    BuiltinKernelId KernelId = static_cast<BuiltinKernelId>(BIKcode);
    if (BIKcode < 0 || BIKcode > POCL_CDBI_LAST) {
      POCL_MSG_ERR("Synapse: Unknown Kernel ID (%s) given\n", Token.c_str());
      continue;
    }

    bool Found = false;
    for (size_t i = 0; i < BIKERNELS; ++i) {
      if (pocl_BIDescriptors[i].KernelId == KernelId) {
        if (D->SupportedList.size() > 0)
          D->SupportedList += ";";
        D->SupportedList += pocl_BIDescriptors[i].name;
        D->SupportedKernels.insert(&pocl_BIDescriptors[i]);
        Found = true;
        break;
      }
    }

    if (!Found) {
      POCL_MSG_ERR("Synapse: Unknown Kernel ID (%s) given, ignoring\n",
                   Token.c_str());
    }
  }

  dev->builtin_kernel_list = strdup(D->SupportedList.c_str());
  dev->num_builtin_kernels = D->SupportedKernels.size();

  pocl_setup_builtin_kernels_with_version(dev);

  POCL_MSG_PRINT_SYNAPSE(
      "accelerator no %u with %zu builtin kernels: (%s)\n",
      j, D->SupportedKernels.size(), dev->builtin_kernel_list);

  POCL_INIT_COND(D->WakeupCond);
  POCL_INIT_LOCK(D->WakeupLock);
  D->WorkQueue = NULL;
  D->ExitRequested = false;
  POCL_CREATE_THREAD(D->DriverThread, &synapseDriverThread, (void*)dev);

  POCL_MSG_PRINT_SYNAPSE("Custom device %d initialized \n", j);

  return CL_SUCCESS;
}

cl_int pocl_synapse_uninit(unsigned j, cl_device_id device) {
  POCL_MSG_PRINT_SYNAPSE("uninit called\n");

  SynapseDeviceData *D = (SynapseDeviceData *)device->data;

  POCL_LOCK(D->WakeupLock);
  D->ExitRequested = true;
  POCL_SIGNAL_COND(D->WakeupCond);
  POCL_UNLOCK(D->WakeupLock);
  POCL_JOIN_THREAD(D->DriverThread);

  POCL_DESTROY_COND(D->WakeupCond);
  POCL_DESTROY_LOCK(D->WakeupLock);

/*
  if (D->ComputeStream) {
    synStreamDestroy(D->ComputeStream);
  }
  if (D->Dev2HostStream) {
    synStreamDestroy(D->Dev2HostStream);
  }
  if (D->Host2DevStream) {
    synStreamDestroy(D->Host2DevStream);
  }
*/
  if (D->DefaultStream) {
    synStreamDestroy(D->DefaultStream);
  }

  delete D;
  --AcquiredDevs;
  if (AcquiredDevs == 0) {
    synDestroy();
  }

  return CL_SUCCESS;
}


//*************************************************************************
//*************************************************************************
//*************************************************************************
//*************************************************************************

void pocl_synapse_write(void *data, const void *__restrict__ src_host_ptr,
                      pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                      size_t offset, size_t size) {
  SynapseDeviceData *D = (SynapseDeviceData *)data;

  char* Src = (char*)src_host_ptr + offset;
  char* Dst = (char*)dst_mem_id->mem_ptr + offset;

  synStatus err = synMemCopyAsync(D->DefaultStream,
                                  (uint64_t)Src, size,
                                  (uint64_t)Dst, HOST_TO_DRAM);
}

void pocl_synapse_read(void *data, void *__restrict__ dst_host_ptr,
                     pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                     size_t offset, size_t size) {
  SynapseDeviceData *D = (SynapseDeviceData *)data;

  char* Src = (char*)src_mem_id->mem_ptr + offset;
  char* Dst = (char*)dst_host_ptr + offset;

  synStatus err = synMemCopyAsync(D->DefaultStream,
                                  (uint64_t)Src, size,
                                  (uint64_t)Dst, DRAM_TO_HOST);

}

void pocl_synapse_copy(void *data, pocl_mem_identifier *dst_mem_id,
                     cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                     cl_mem src_buf, size_t dst_offset, size_t src_offset,
                     size_t size) {
  SynapseDeviceData *D = (SynapseDeviceData *)data;

  char* Src = (char*)src_mem_id->mem_ptr + src_offset;
  char* Dst = (char*)dst_mem_id->mem_ptr + dst_offset;

  if (Src == Dst) {
    return;
  }

  synStatus err = synMemCopyAsync(D->DefaultStream,
                                  (uint64_t)Src, size,
                                  (uint64_t)Dst, DRAM_TO_DRAM);

}

void pocl_synapse_memfill(void *data, pocl_mem_identifier *dst_mem_id,
                      cl_mem dst_buf, size_t size, size_t offset,
                      const void *__restrict__ pattern, size_t pattern_size) {
  if (pattern_size > 4) {
    POCL_ABORT_UNIMPLEMENTED("synapse memfill with pattern size > 4");
  }

  SynapseDeviceData *D = (SynapseDeviceData *)data;
  synStatus err;
  char* Dst = (char*)dst_mem_id->mem_ptr + offset;
  uint8_t Pat8 = ((uint8_t*)pattern)[0];
  uint16_t Pat16 = ((uint16_t*)pattern)[0];
  uint32_t Pat32 = ((uint32_t*)pattern)[0];

  switch (pattern_size) {
    case 1:  err = synMemsetD8Async((uint64_t)Dst, Pat8, size, D->DefaultStream);
    case 2:  err = synMemsetD16Async((uint64_t)Dst, Pat16, size/2, D->DefaultStream);
    case 4:  err = synMemsetD32Async((uint64_t)Dst, Pat32, size/4, D->DefaultStream);
    default: POCL_MSG_ERR("unknown pattern size\n");
  }
}


cl_int pocl_synapse_alloc_mem_obj(cl_device_id device, cl_mem mem_obj,
                                void *host_ptr) {

  SynapseDeviceData *D = (SynapseDeviceData *)device->data;

  /* synapse driver doesn't preallocate */
  if ((mem_obj->flags & CL_MEM_ALLOC_HOST_PTR) &&
      (mem_obj->mem_host_ptr == NULL))
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  void *HostTemp = nullptr;
  uint64_t DevTemp = 0;
  synStatus err;
  pocl_mem_identifier *P = &mem_obj->device_ptrs[device->global_mem_id];

  err = synHostMalloc(D->DevID, mem_obj->size, 0, (void**)&HostTemp);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synHostMalloc() FAILED\n");
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }
  err = synDeviceMalloc(D->DevID, mem_obj->size, 0, 0, &DevTemp);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synDeviceMalloc() FAILED\n");
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  P->mem_ptr = (void*)DevTemp;
  P->extra_ptr = HostTemp;
  P->extra = 0;
  P->version = 0;
  P->is_pinned = 0;
  P->device_addr = 0;

  return CL_SUCCESS;
}


void pocl_synapse_free(cl_device_id device, cl_mem mem) {

  pocl_mem_identifier *P = &mem->device_ptrs[device->global_mem_id];
  SynapseDeviceData *D = (SynapseDeviceData *)device->data;

  synStatus err;
  err = synHostFree(D->DevID, P->extra_ptr, 0);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synHostFree() FAILED\n");
  }
  err = synDeviceFree(D->DevID, (uint64_t)P->mem_ptr, 0);
  if (err != synSuccess) {
    POCL_MSG_ERR("Synapse: synDeviceFree() FAILED\n");
  }

  P->extra_ptr = 0;
  P->mem_ptr = NULL;
  P->version = 0;
}

cl_int pocl_synapse_map_mem(void *data, pocl_mem_identifier *src_mem_id,
                          cl_mem src_buf, mem_mapping_t *map) {

  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  /* Synch the device global region to the host memory. */
  pocl_synapse_read(data, map->host_ptr, src_mem_id, src_buf, map->offset,
                    map->size);

  return CL_SUCCESS;
}

cl_int pocl_synapse_unmap_mem(void *data, pocl_mem_identifier *dst_mem_id,
                            cl_mem dst_buf, mem_mapping_t *map) {

  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;

  /* Synch the host memory to the device global region. */
  pocl_synapse_write(data, map->host_ptr, dst_mem_id, dst_buf, map->offset,
                     map->size);

  return CL_SUCCESS;
}

cl_int
pocl_synapse_get_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                             cl_mem mem, mem_mapping_t *map)
{
  SynapseDeviceData *D = (SynapseDeviceData *)data;
  /* assume buffer is allocated */
  assert (mem_id->mem_ptr != NULL);
  assert (mem_id->extra_ptr != NULL);
  assert (mem->size > 0);
  assert (map->size > 0);

  if (mem->flags & CL_MEM_USE_HOST_PTR)
    map->host_ptr = (char *)mem->mem_host_ptr + map->offset;
  else
    map->host_ptr = (char *)mem_id->extra_ptr + map->offset;
  /* POCL_MSG_WARN ("Synapse map HOST_PTR: %p | SIZE %zu | OFFS %zu | DEV PTR: %p \n",
                     map->host_ptr, map->size, map->offset, mem_id->mem_ptr); */
  assert (map->host_ptr);
  return CL_SUCCESS;
}

cl_int
pocl_synapse_free_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                              cl_mem mem, mem_mapping_t *map)
{
  SynapseDeviceData *D = (SynapseDeviceData *)data;
  map->host_ptr = NULL;
  return CL_SUCCESS;
}


//*************************************************************************
//*************************************************************************
//*************************************************************************
//*************************************************************************


/*
void pocl_synapse_update_event(cl_device_id device, cl_event event) {
  SynapseDeviceData *D = (SynapseDeviceData *)device->data;
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  union {
    struct {
      uint32_t a;
      uint32_t b;
    } u32;
    uint64_t u64;
  } timestamp;

  if ((event->queue->properties & CL_QUEUE_PROFILING_ENABLE) &&
      (event->command_type == CL_COMMAND_NDRANGE_KERNEL)) {

    if (event->status <= CL_COMPLETE) {
      assert(ed);
      size_t commandMetaAddress = ed->chunk->start_address;
      assert(commandMetaAddress);
      commandMetaAddress -= D->Dev->DataMemory->PhysAddress();

      timestamp.u32.a = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, start_timestamp));
      timestamp.u32.b = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, start_timestamp) +
          sizeof(uint32_t));
      if (timestamp.u64 > 0)
        event->time_start = timestamp.u64;

      timestamp.u32.a = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, finish_timestamp));
      timestamp.u32.b = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, finish_timestamp) +
          sizeof(uint32_t));
      if (timestamp.u64 > 0)
        event->time_end = timestamp.u64;

      // recalculation of timestamps to host clock
      if (D->Dev->HasHardwareClock && D->Dev->HwClockFrequency > 0) {
        double NsPerClock =
            (double)1000000000.0 / (double)D->Dev->HwClockFrequency;

        double StartNs = (double)event->time_start * NsPerClock;
        event->time_start = D->Dev->HwClockStart + (uint64_t)StartNs;

        double EndNs = (double)event->time_end * NsPerClock;
        event->time_end = D->Dev->HwClockStart + (uint64_t)EndNs;
      }

      if (device->device_side_printf) {
        chunk_info_t *PrintfBufferChunk = (chunk_info_t *)D->PrintfBuffer;
        assert(PrintfBufferChunk);
        chunk_info_t *PrintfPositionChunk = (chunk_info_t *)D->PrintfPosition;
        assert(PrintfPositionChunk);
        unsigned position =
            D->Dev->DataMemory->Read32(PrintfPositionChunk->start_address -
                                       D->Dev->DataMemory->PhysAddress());
        POCL_MSG_PRINT_ALMAIF(
            "Device wrote %u bytes to stdout. Printing them now:\n", position);
        if (position > 0) {
          char *tmp_printf_buf = (char *)malloc(position);
          D->Dev->DataMemory->CopyFromMMAP(
              tmp_printf_buf, PrintfBufferChunk->start_address, position);
          D->Dev->DataMemory->Write32(PrintfPositionChunk->start_address -
                                          D->Dev->DataMemory->PhysAddress(),
                                      0);
          write(STDOUT_FILENO, tmp_printf_buf, position);
          free(tmp_printf_buf);
        }
      }
    }
  }
}

static void scheduleCommands(SynapseDeviceData &D) {

  _cl_command_node *Node;
  // Execute commands from ready list.
  while ((Node = D.ReadyList)) {
    assert(Node->sync.event.event->status == CL_SUBMITTED);

    if (Node->type == CL_COMMAND_NDRANGE_KERNEL) {
      pocl_update_event_running(Node->sync.event.event);

      submit_and_barrier((SynapseDeviceData *)Node->device->data, Node);
      submit_kernel_packet((SynapseDeviceData *)Node->device->data, Node);

      POCL_LOCK(runningLock);
      CDL_DELETE(D.ReadyList, Node);
      DL_PREPEND(runningList, Node);
      POCL_UNLOCK(runningLock);
    } else {
      assert(pocl_command_is_ready(Node->sync.event.event));

      CDL_DELETE(D.ReadyList, Node);
      POCL_UNLOCK(D.CommandListLock);
      pocl_exec_command(Node);
      POCL_LOCK(D.CommandListLock);
    }
  }

  return;
}

void pocl_synapse_submit(_cl_command_node *Node, cl_command_queue) {

  Node->ready = 1;

  struct SynapseDeviceData *D = (SynapseDeviceData *)Node->device->data;
  cl_event E = Node->sync.event.event;

  if (E->data == nullptr) {
    E->data = calloc(1, sizeof(almaif_event_data_t));
    assert(E->data && "event data allocation failed");
    almaif_event_data_t *ed = (almaif_event_data_t *)E->data;
    POCL_INIT_COND(ed->event_cond);
  }

  if ((Node->type == CL_COMMAND_NDRANGE_KERNEL) &&
      only_custom_device_events_left(E)) {
    pocl_update_event_submitted(E);

    POCL_UNLOCK_OBJ(E);
    pocl_update_event_running(E);

    submit_and_barrier(D, Node);
    submit_kernel_packet(D, Node);

    POCL_LOCK(runningLock);
    DL_PREPEND(runningList, Node);
    POCL_UNLOCK(runningLock);
  } else {
    POCL_LOCK(D->CommandListLock);
    pocl_command_push(Node, &D->ReadyList, &D->CommandList);

    POCL_UNLOCK_OBJ(E);
    scheduleCommands(*D);
    POCL_UNLOCK(D->CommandListLock);
  }

  return;
}
void pocl_synapse_join(cl_device_id device, cl_command_queue cq) {

  struct SynapseDeviceData *D = (SynapseDeviceData *)device->data;
  POCL_LOCK(D->CommandListLock);
  while (D->CommandList || D->ReadyList) {
    scheduleCommands(*D);
    POCL_UNLOCK(D->CommandListLock);
    usleep(ALMAIF_DRIVER_SLEEP);
    POCL_LOCK(D->CommandListLock);
  }

  POCL_UNLOCK(D->CommandListLock);

  POCL_LOCK_OBJ(cq);
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  assert(cq_cond);
  while (1) {
    if (cq->command_count == 0) {
      POCL_UNLOCK_OBJ(cq);
      return;
    } else {
      POCL_WAIT_COND(*cq_cond, cq->pocl_lock);
    }
  }
  return;
}

void pocl_synapse_notify(cl_device_id Device, cl_event Event, cl_event Finished) {

  struct SynapseDeviceData &D = *(SynapseDeviceData *)Device->data;

  _cl_command_node *volatile Node = Event->command;

  if (Finished->status < CL_COMPLETE) {
    pocl_update_event_failed(Event);
    return;
  }

  if (!Node->ready)
    return;

  if (Event->command->type != CL_COMMAND_NDRANGE_KERNEL) {
    if (pocl_command_is_ready(Event)) {
      if (Event->status == CL_QUEUED) {
        pocl_update_event_submitted(Event);
        POCL_LOCK(D.CommandListLock);
        CDL_DELETE(D.CommandList, Node);
        CDL_PREPEND(D.ReadyList, Node);

        POCL_UNLOCK_OBJ(Event);
        scheduleCommands(D);
        POCL_LOCK_OBJ(Event);

        POCL_UNLOCK(D.CommandListLock);
      }
    }
  } else {
    if (only_custom_device_events_left(Event)) {
      pocl_update_event_submitted(Event);
      POCL_LOCK(D.CommandListLock);
      CDL_DELETE(D.CommandList, Node);
      CDL_PREPEND(D.ReadyList, Node);

      POCL_UNLOCK_OBJ(Event);
      scheduleCommands(D);
      POCL_LOCK_OBJ(Event);

      POCL_UNLOCK(D.CommandListLock);
    }
  }
}

void scheduleNDRange(SynapseDeviceData *data, _cl_command_node *cmd, size_t arg_size,
                     void *arguments) {
  _cl_command_run *run = &cmd->command.run;
  cl_kernel k = run->kernel;
  cl_program p = k->program;
  cl_event e = cmd->sync.event.event;
  almaif_event_data_t *event_data = (almaif_event_data_t *)e->data;
  int32_t kernelID = -1;
  bool SanitizeKernelName = false;

  for (auto supportedKernel : data->SupportedKernels) {
    if (strcmp(supportedKernel->name, k->name) == 0) {
      kernelID = (int32_t)supportedKernel->KernelId;
      // builtin kernels that come from tce_kernels.cl need compiling
      if (p->num_builtin_kernels > 0 && p->source) {
        POCL_MSG_PRINT_ALMAIF(
            "almaif: builtin kernel with source, needs compiling\n");
        kernelID = -1;
        SanitizeKernelName = true;
      }
      break;
    }
  }
#ifdef HAVE_DBDEVICE
  if (data->Dev->isDBDevice()) {
    ((DBDevice *)(data->Dev))
        ->programBIKernelBitstream((BuiltinKernelId)kernelID);
    ((DBDevice *)(data->Dev))
        ->programBIKernelFirmware((BuiltinKernelId)kernelID);
  }
#endif

  if (kernelID == -1) {
    if (data->compilationData == NULL) {
      POCL_ABORT("almaif: scheduled an NDRange with unsupported kernel\n");
    } else {
      POCL_MSG_PRINT_ALMAIF("almaif: compiling kernel\n");
      char *SavedName = nullptr;
      if (SanitizeKernelName)
        pocl_sanitize_builtin_kernel_name(k, &SavedName);

      pocl_synapse_compile_kernel(cmd, k, cmd->device, 1);

      if (SanitizeKernelName)
        pocl_restore_builtin_kernel_name(k, SavedName);
    }
  }

  // Additional space for a signal
  size_t extraAlloc = sizeof(struct CommandMetadata);
  chunk_info_t *chunk = pocl_alloc_buffer_from_region(data->Dev->AllocRegions,
                                                      arg_size + extraAlloc);
  assert(chunk && "Failed to allocate signal/argument buffer");

  POCL_MSG_PRINT_ALMAIF("almaif: allocated 0x%zx bytes for signal/arguments "
                       "from 0x%zx\n",
                       arg_size + extraAlloc, chunk->start_address);
  assert(event_data);
  assert(event_data->chunk == NULL);
  event_data->chunk = chunk;

  size_t commandMetaAddress = chunk->start_address;
  size_t signalAddress =
      commandMetaAddress + offsetof(CommandMetadata, completion_signal);
  size_t argsAddress = chunk->start_address + sizeof(struct CommandMetadata);
  POCL_MSG_PRINT_ALMAIF("Signal address=0x%zx\n", signalAddress);
  // clear the timestamps and initial signal value
  for (unsigned offset = 0; offset < sizeof(CommandMetadata); offset += 4)
    data->Dev->DataMemory->Write32(
        commandMetaAddress - data->Dev->DataMemory->PhysAddress() + offset, 0);
  if (cmd->device->device_side_printf) {
    data->Dev->DataMemory->Write32(
        commandMetaAddress - data->Dev->DataMemory->PhysAddress() +
            offsetof(CommandMetadata, reserved0),
        ((chunk_info_t *)data->PrintfBuffer)->start_address);
    data->Dev->DataMemory->Write32(commandMetaAddress -
                                       data->Dev->DataMemory->PhysAddress() +
                                       offsetof(CommandMetadata, reserved1),
                                   cmd->device->printf_buffer_size);

    data->Dev->DataMemory->Write32(
        commandMetaAddress - data->Dev->DataMemory->PhysAddress() +
            offsetof(CommandMetadata, reserved1) + 4,
        ((chunk_info_t *)data->PrintfPosition)->start_address);
  }

  // Set arguments
  data->Dev->DataMemory->CopyToMMAP(argsAddress, arguments, arg_size);

  struct AQLDispatchPacket packet = {};

  packet.header = AQL_PACKET_INVALID;
  packet.dimensions = run->pc.work_dim; // number of dimensions

  packet.workgroup_size_x = run->pc.local_size[0];
  packet.workgroup_size_y = run->pc.local_size[1];
  packet.workgroup_size_z = run->pc.local_size[2];

  pocl_context32 pc;

  if (kernelID != -1) {
    packet.grid_size_x = run->pc.local_size[0] * run->pc.num_groups[0];
    packet.grid_size_y = run->pc.local_size[1] * run->pc.num_groups[1];
    packet.grid_size_z = run->pc.local_size[2] * run->pc.num_groups[2];
    packet.kernel_object = kernelID;
  } else {

    // Compilation needs pocl_context struct, create it, copy it to device and
    // pass the pointer to it in the 'reserved' slot of AQL kernel dispatch
    // packet.
    pc.work_dim = run->pc.work_dim;
    pc.local_size[0] = run->pc.local_size[0];
    pc.local_size[1] = run->pc.local_size[1];
    pc.local_size[2] = run->pc.local_size[2];
    pc.num_groups[0] = run->pc.num_groups[0];
    pc.num_groups[1] = run->pc.num_groups[1];
    pc.num_groups[2] = run->pc.num_groups[2];
    pc.global_offset[0] = run->pc.global_offset[0];
    pc.global_offset[1] = run->pc.global_offset[1];
    pc.global_offset[2] = run->pc.global_offset[2];
    pc.global_var_buffer = 0;

    if (cmd->device->device_side_printf) {
      pc.printf_buffer = ((chunk_info_t *)data->PrintfBuffer)->start_address;
      pc.printf_buffer_capacity = cmd->device->printf_buffer_size;
      assert(pc.printf_buffer_capacity);

      pc.printf_buffer_position =
          ((chunk_info_t *)data->PrintfPosition)->start_address;
      POCL_MSG_PRINT_ALMAIF(
          "Device side printf buffer=%d, position: %d and capacity %d \n",
          pc.printf_buffer, pc.printf_buffer_position,
          pc.printf_buffer_capacity);

      data->Dev->DataMemory->Write32(
          pc.printf_buffer_position - data->Dev->DataMemory->PhysAddress(), 0);
    }

    size_t pc_start_addr = data->compilationData->pocl_context->start_address;
    data->Dev->DataMemory->CopyToMMAP(pc_start_addr, &pc,
                                      sizeof(pocl_context32));

    if (data->Dev->RelativeAddressing) {
      pc_start_addr -= data->Dev->DataMemory->PhysAddress();
    }

    packet.reserved = pc_start_addr;

    almaif_kernel_data_t *kd =
        (almaif_kernel_data_t *)run->kernel->data[cmd->program_device_i];
    packet.kernel_object = kd->kernel_address;

    POCL_MSG_PRINT_ALMAIF("Kernel addresss=0x%" PRIu32 "\n", kd->kernel_address);
  }

  if (data->Dev->RelativeAddressing) {
    packet.kernarg_address = argsAddress - data->Dev->DataMemory->PhysAddress();
    packet.command_meta_address =
        commandMetaAddress - data->Dev->DataMemory->PhysAddress();
  } else {
    packet.kernarg_address = argsAddress;
    packet.command_meta_address = commandMetaAddress;
  }

  POCL_MSG_PRINT_ALMAIF("ArgsAddress=0x%" PRIx64 " CommandMetaAddress=0x%" PRIx64
                       " \n",
                       packet.kernarg_address, packet.command_meta_address);

  POCL_LOCK(data->AQLQueueLock);
  uint32_t queue_length = data->Dev->CQMemory->Size() / AQL_PACKET_LENGTH - 1;

  uint32_t write_iter = data->Dev->CQMemory->Read32(ALMAIF_CQ_WRITE);
  uint32_t read_iter = data->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
  while (write_iter >= read_iter + queue_length) {
    POCL_MSG_PRINT_ALMAIF("write_iter=%u, read_iter=%u length=%u", write_iter,
                          read_iter, queue_length);
    usleep(ALMAIF_DRIVER_SLEEP);
    read_iter = data->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
#ifdef ALMAIF_DUMP_MEMORY
    POCL_MSG_PRINT_ALMAIF("WAITING FOR CQMEMORY TO EMPTY DUMP\n");
    data->Dev->printMemoryDump();
#endif
  }
  uint32_t packet_loc =
      (write_iter % queue_length) * AQL_PACKET_LENGTH + AQL_PACKET_LENGTH;
  data->Dev->CQMemory->CopyToMMAP(
      packet_loc + data->Dev->CQMemory->PhysAddress(), &packet, 64);

#ifdef ALMAIF_DUMP_MEMORY
  POCL_MSG_PRINT_ALMAIF("PRELAUNCH MEMORY DUMP\n");
  data->Dev->printMemoryDump();
#endif

  // finally, set header as not-invalid
  data->Dev->CQMemory->Write16(packet_loc, (1 << AQL_PACKET_KERNEL_DISPATCH) |
                                               AQL_PACKET_BARRIER);

  POCL_MSG_PRINT_ALMAIF(
      "almaif: Handed off a packet for execution, write iter=%u\n", write_iter);
  // Increment queue index
  data->Dev->CQMemory->Write32(ALMAIF_CQ_WRITE, write_iter + 1);

  POCL_UNLOCK(data->AQLQueueLock);

  if (kernelID == -1 && data->compilationData &&
      data->compilationData->produce_standalone_program) {
    data->compilationData->produce_standalone_program(data, cmd, &pc, arg_size,
                                                      arguments);
  }
}

bool isEventDone(SynapseDeviceData *data, cl_event event) {

  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  if (ed->chunk->start_address == 0)
    return false;

  size_t commandMetaAddress = ed->chunk->start_address;
  assert(commandMetaAddress);
  size_t signalAddress =
      commandMetaAddress + offsetof(CommandMetadata, completion_signal);
  signalAddress -= data->Dev->DataMemory->PhysAddress();

  uint32_t status = data->Dev->DataMemory->Read32(signalAddress);

  if (status == 1) {
    POCL_MSG_PRINT_ALMAIF("Event %" PRIu64
                         " done, completion signal address=%zx, value=%u\n",
                         event->id, signalAddress, status);
  }

  return (status == 1);
}

void pocl_synapse_wait_event(cl_device_id device, cl_event event) {
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;

  POCL_LOCK_OBJ(event);
  while (event->status > CL_COMPLETE) {
    POCL_WAIT_COND(ed->event_cond, event->pocl_lock);
  }
  POCL_UNLOCK_OBJ(event);
}

void pocl_synapse_notify_cmdq_finished(cl_command_queue cq) {
  // must be called with CQ already locked.
   // this must be a broadcast since there could be multiple
   // user threads waiting on the same command queue
   // in pthread_scheduler_wait_cq().
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  PTHREAD_CHECK(pthread_cond_broadcast(cq_cond));
}

void pocl_synapse_notify_event_finished(cl_event event) {
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  POCL_BROADCAST_COND(ed->event_cond);

  // this is a hack required b/c pocld does not release events,
  // the "pocl_synapse_free_event_data" is not called, and because
  // almaif allocates memory from device globalmem for signals,
  // the device eventually runs out of memory.
  if (event->command_type == CL_COMMAND_NDRANGE_KERNEL && ed->chunk != NULL) {
    pocl_free_chunk((chunk_info_t *)ed->chunk);
    ed->chunk = NULL;
  }
}

int pocl_synapse_init_queue(cl_device_id device, cl_command_queue queue) {
  queue->data = malloc(sizeof(pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  POCL_INIT_COND(*cond);
  return CL_SUCCESS;
}

int pocl_synapse_free_queue(cl_device_id device, cl_command_queue queue) {
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  POCL_DESTROY_COND(*cond);
  POCL_MEM_FREE(queue->data);
  return CL_SUCCESS;
}

void submit_and_barrier(SynapseDeviceData *D, _cl_command_node *cmd) {

  event_node *dep_event = cmd->sync.event.event->wait_list;
  if (dep_event == NULL) {
    POCL_MSG_PRINT_ALMAIF("Almaif: no events to wait for\n");
    return;
  }

  bool all_done = false;
  while (!all_done) {
    struct AQLAndPacket packet = {};
    memset(&packet, 0, sizeof(AQLAndPacket));
    packet.header = AQL_PACKET_INVALID;
    int i;
    for (i = 0; i < AQL_MAX_SIGNAL_COUNT; i++) {
      almaif_event_data_t *dep_ed = (almaif_event_data_t *)dep_event->event->data;
      assert(dep_ed);
      if (dep_ed->chunk) {
        packet.dep_signals[i] = dep_ed->chunk->start_address;
        POCL_MSG_PRINT_ALMAIF(
            "Creating AND barrier depending on signal id=%" PRIu64
            " at address %" PRIu64 " \n",
            dep_event->event->id, packet.dep_signals[i]);
      }
      dep_event = dep_event->next;
      if (dep_event == NULL) {
        all_done = true;
        break;
      }
    }
    packet.signal_count = i + 1;

    POCL_LOCK(D->AQLQueueLock);
    uint32_t queue_length = D->Dev->CQMemory->Size() / AQL_PACKET_LENGTH - 1;

    uint32_t write_iter = D->Dev->CQMemory->Read32(ALMAIF_CQ_WRITE);
    uint32_t read_iter = D->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
    while (write_iter >= read_iter + queue_length) {
      POCL_MSG_PRINT_ALMAIF("write_iter=%u, read_iter=%u length=%u", write_iter,
                            read_iter, queue_length);
      read_iter = D->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
      usleep(ALMAIF_DRIVER_SLEEP);
    }
    uint32_t packet_loc =
        (write_iter % queue_length) * AQL_PACKET_LENGTH + AQL_PACKET_LENGTH;
    D->Dev->CQMemory->CopyToMMAP(packet_loc + D->Dev->CQMemory->PhysAddress(),
                                 &packet, 64);

    D->Dev->CQMemory->Write16(packet_loc, (1 << AQL_PACKET_BARRIER_AND) |
                                              AQL_PACKET_BARRIER);

    POCL_MSG_PRINT_ALMAIF("almaif: Handed off and barrier, write iter=%u\n",
                         write_iter);
    // Increment queue index
    D->Dev->CQMemory->Write32(ALMAIF_CQ_WRITE, write_iter + 1);

    POCL_UNLOCK(D->AQLQueueLock);
  }
}

void pocl_synapse_run(void *data, _cl_command_node *cmd) {}

void submit_kernel_packet(SynapseDeviceData *D, _cl_command_node *cmd) {
  struct pocl_argument *al;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0 || pc->num_groups[2] == 0)
    return;

  // First pass to figure out total argument size
  size_t arg_size = 0;
  for (i = 0; i < meta->num_args; ++i) {
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      arg_size += D->Dev->PointerSize;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_PIPE) {
      arg_size += 4;
    } else {
      al = &(cmd->command.run.arguments[i]);
      arg_size += al->size;
    }
  }
  void *arguments = malloc(arg_size);
  char *current_arg = (char *)arguments;
  // TODO: Refactor this to a helper function (the argbuffer ABI).
  // Process the kernel arguments. Convert the opaque buffer
  // pointers to real device pointers, allocate dynamic local
  // memory buffers, etc.
  for (i = 0; i < meta->num_args; ++i) {
    al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i]))
      // No kernels with local args at the moment, should not end up here
      POCL_ABORT_UNIMPLEMENTED("almaif: local arguments");
    else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      // It's legal to pass a NULL pointer to clSetKernelArguments. In
      // that case we must pass the same NULL forward to the kernel.
      // Otherwise, the user must have created a buffer with per device
      // pointers stored in the cl_mem.
      if (al->value == NULL) {
        *(size_t *)current_arg = 0;
      } else {
        // almaif doesn't support SVM pointers
        assert(al->is_svm == 0);
        cl_mem m = (*(cl_mem *)(al->value));
        size_t buffer = D->Dev->pointerDeviceOffset(
            &(m->device_ptrs[cmd->device->global_mem_id]));
        buffer += al->offset;
        if (D->Dev->RelativeAddressing) {
          if (D->Dev->DataMemory->isInRange(buffer)) {
            buffer -= D->Dev->DataMemory->PhysAddress();
          } else if (D->Dev->ExternalMemory->isInRange(buffer)) {
            buffer -= D->Dev->ExternalMemory->PhysAddress();
          } else {
            POCL_ABORT("almaif: buffer outside of memory");
          }
        }
        *(size_t *)current_arg = buffer;
      }
      current_arg += D->Dev->PointerSize;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      POCL_ABORT_UNIMPLEMENTED("almaif: image arguments");
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_ABORT_UNIMPLEMENTED("almaif: sampler arguments");
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_PIPE) {
      cl_mem m = (*(cl_mem *)(al->value));
      int *pipe_id_ptr =
          (int *)(m->device_ptrs[cmd->device->global_mem_id].mem_ptr);
      int pipe_id = *pipe_id_ptr;
      *(int *)current_arg = pipe_id;
      POCL_MSG_PRINT_ALMAIF("Setting pipe argument %d to id: %i\n", i, pipe_id);
      current_arg += 4;
    } else {
      memcpy(current_arg, al->value, al->size);
      current_arg += al->size;
    }
  }

  scheduleNDRange(D, cmd, arg_size, arguments);

  POCL_MEM_FREE(cmd->command.run.device_data);
  free(arguments);
}

void pocl_synapse_free_event_data(cl_event event) {
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  if (ed) {
    if (ed->chunk != NULL) {
      pocl_free_chunk((chunk_info_t *)ed->chunk);
    }
    POCL_DESTROY_COND(ed->event_cond);
    POCL_MEM_FREE(event->data);
  }
  event->data = NULL;
}

void *runningThreadFunc(void *) {
  int counter = 0;
  while (!runningJoinRequested) {
    POCL_LOCK(runningLock);
    if (runningList) {
      _cl_command_node *Node = NULL;
      _cl_command_node *tmp = NULL;
      DL_FOREACH_SAFE(runningList, Node, tmp) {
        SynapseDeviceData *AD = (SynapseDeviceData *)Node->device->data;
        if (isEventDone(AD, Node->sync.event.event)) {
          DL_DELETE(runningList, Node);
          cl_event E = Node->sync.event.event;
#ifdef ALMAIF_DUMP_MEMORY
          POCL_MSG_PRINT_ALMAIF("FINAL MEMORY DUMP\n");
          AD->Dev->printMemoryDump();
#endif
          POCL_UNLOCK(runningLock);
          POCL_UPDATE_EVENT_COMPLETE_MSG(E, "Almaif, asynchronous NDRange    ");
          POCL_LOCK(runningLock);
        }

#ifdef ALMAIF_DUMP_MEMORY
        if ((counter % 3) == 0) {
          if (Node->device->device_side_printf) {
            chunk_info_t *PrintfBufferChunk = (chunk_info_t *)AD->PrintfBuffer;
            assert(PrintfBufferChunk);
            chunk_info_t *PrintfPositionChunk =
                (chunk_info_t *)AD->PrintfPosition;
            assert(PrintfPositionChunk);
            unsigned position =
                AD->Dev->DataMemory->Read32(PrintfPositionChunk->start_address -
                                            AD->Dev->DataMemory->PhysAddress());
            POCL_MSG_PRINT_ALMAIF(
                "Device wrote %u bytes to stdout. Printing them now:\n",
                position);
            if (position > 0) {
              char *tmp_printf_buf = (char *)malloc(position);
              AD->Dev->DataMemory->CopyFromMMAP(
                  tmp_printf_buf, PrintfBufferChunk->start_address, position);
              write(STDOUT_FILENO, tmp_printf_buf, position);
              free(tmp_printf_buf);
            }
          }
        } else {
          uint32_t pc = AD->Dev->ControlMemory->Read32(ALMAIF_STATUS_REG_PC);
          uint64_t cc =
              AD->Dev->ControlMemory->Read64(ALMAIF_STATUS_REG_CC_LOW);
          uint64_t sc =
              AD->Dev->ControlMemory->Read64(ALMAIF_STATUS_REG_SC_LOW);
          POCL_MSG_PRINT_ALMAIF(
              "PC:%" PRId32 " CC:%" PRId64 " SC:%" PRId64 "\n", pc, cc, sc);

          POCL_MSG_PRINT_ALMAIF("RUNNING MEMORY DUMP\n");
          AD->Dev->printMemoryDump();
        }
#endif
        counter++;
      }
    }
    POCL_UNLOCK(runningLock);
    usleep(ALMAIF_DRIVER_SLEEP);
  }
  return NULL;
}

void pocl_synapse_copy_rect(void *data, pocl_mem_identifier *dst_mem_id,
                          cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                          cl_mem src_buf,
                          const size_t *__restrict__ const dst_origin,
                          const size_t *__restrict__ const src_origin,
                          const size_t *__restrict__ const region,
                          size_t const dst_row_pitch,
                          size_t const dst_slice_pitch,
                          size_t const src_row_pitch,
                          size_t const src_slice_pitch) {
  SynapseDeviceData *d = (SynapseDeviceData *)data;

  size_t src_offset = src_origin[0] + src_row_pitch * src_origin[1] +
                      src_slice_pitch * src_origin[2];
  size_t dst_offset = dst_origin[0] + dst_row_pitch * dst_origin[1] +
                      dst_slice_pitch * dst_origin[2];

  size_t j, k, i;

  // TODO: handle overlaping regions

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      for (i = 0; i < region[0]; i++) {
        char val;
        d->Dev->readDataFromDevice(&val, src_mem_id, 1,
                                   src_offset + src_row_pitch * j +
                                       src_slice_pitch * k + i);
        d->Dev->writeDataToDevice(dst_mem_id, &val, 1,
                                  dst_offset + dst_row_pitch * j +
                                      dst_slice_pitch * k + i);
      }
}

void pocl_synapse_write_rect(void *data, const void *__restrict__ src_host_ptr,
                           pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                           const size_t *__restrict__ const buffer_origin,
                           const size_t *__restrict__ const host_origin,
                           const size_t *__restrict__ const region,
                           size_t const buffer_row_pitch,
                           size_t const buffer_slice_pitch,
                           size_t const host_row_pitch,
                           size_t const host_slice_pitch) {
  SynapseDeviceData *d = (SynapseDeviceData *)data;
  size_t adjusted_dst_offset = buffer_origin[0] +
                               buffer_row_pitch * buffer_origin[1] +
                               buffer_slice_pitch * buffer_origin[2];

  char const *__restrict__ const adjusted_host_ptr =
      (char const *)src_host_ptr + host_origin[0] +
      host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];

  size_t j, k;

  // TODO: handle overlapping regions

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j) {
      size_t s_offset = host_row_pitch * j + host_slice_pitch * k;

      size_t d_offset = buffer_row_pitch * j + buffer_slice_pitch * k;

      d->Dev->writeDataToDevice(dst_mem_id, adjusted_host_ptr + s_offset,
                                region[0], adjusted_dst_offset + d_offset);
    }
}

void pocl_synapse_read_rect(void *data, void *__restrict__ dst_host_ptr,
                          pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                          const size_t *__restrict__ const buffer_origin,
                          const size_t *__restrict__ const host_origin,
                          const size_t *__restrict__ const region,
                          size_t const buffer_row_pitch,
                          size_t const buffer_slice_pitch,
                          size_t const host_row_pitch,
                          size_t const host_slice_pitch) {
  SynapseDeviceData *d = (SynapseDeviceData *)data;
  size_t adjusted_src_offset = buffer_origin[0] +
                               buffer_row_pitch * buffer_origin[1] +
                               buffer_slice_pitch * buffer_origin[2];

  char *__restrict__ const adjusted_host_ptr =
      (char *)dst_host_ptr + host_origin[0] + host_row_pitch * host_origin[1] +
      host_slice_pitch * host_origin[2];

  size_t j, k;

  // TODO: handle overlaping regions

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j) {
      size_t d_offset = host_row_pitch * j + host_slice_pitch * k;
      size_t s_offset = buffer_row_pitch * j + buffer_slice_pitch * k;
      d->Dev->readDataFromDevice(adjusted_host_ptr + d_offset, src_mem_id,
                                 region[0], adjusted_src_offset + s_offset);
    }
}
*/

/********************************************************************/

static void
synapse_push_command (cl_device_id Dev, _cl_command_node *Cmd)
{
  SynapseDeviceData *D = (SynapseDeviceData *)Dev->data;
  POCL_LOCK (D->WakeupLock);
  DL_APPEND (D->WorkQueue, Cmd);
  POCL_SIGNAL_COND (D->WakeupCond);
  POCL_FAST_UNLOCK (D->WakeupLock);
}

void
pocl_synapse_submit (_cl_command_node *Node, cl_command_queue CQ)
{
  Node->ready = 1;
  if (pocl_command_is_ready (Node->sync.event.event))
    {
      pocl_update_event_submitted (Node->sync.event.event);
      synapse_push_command (CQ->device, Node);
    }
  POCL_UNLOCK_OBJ (Node->sync.event.event);
  return;
}

int
pocl_synapse_init_queue (cl_device_id Dev, cl_command_queue Queue)
{
  SynapseDeviceData *D = (SynapseDeviceData *)Dev->data;
  SynapseCommandQueue *CQ = new SynapseCommandQueue(D->DevID);
  if (CQ == nullptr)
    return CL_OUT_OF_HOST_MEMORY;
  if (!CQ->init())
    return CL_INVALID_DEVICE;

  Queue->data = CQ;
  return CL_SUCCESS;
}

int
pocl_synapse_free_queue (cl_device_id Dev, cl_command_queue Queue)
{
  SynapseCommandQueue *CQ = (SynapseCommandQueue *)Queue->data;
  delete CQ;
  Queue->data = nullptr;
  return CL_SUCCESS;
}

void
pocl_synapse_notify_cmdq_finished (cl_command_queue Queue)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  SynapseCommandQueue *CQ = (SynapseCommandQueue *)Queue->data;
  CQ->broadcast();
}

void
pocl_synapse_notify_event_finished (cl_event Event)
{
  SynapseCond *Cond = (SynapseCond *)Event->data;
  Cond->broadcast();
}


void
pocl_synapse_join (cl_device_id Device, cl_command_queue Queue)
{
  POCL_LOCK_OBJ (Queue);
  SynapseCommandQueue *CQ = (SynapseCommandQueue *)Queue->data;
  while (1)
    {
      if (Queue->command_count == 0)
        {
          POCL_UNLOCK_OBJ (Queue);
          return;
        }
      else
        {
          CQ->wait(Queue->pocl_lock);
        }
    }
  return;
}

void
pocl_synapse_flush (cl_device_id device, cl_command_queue cq)
{
}

void
pocl_synapse_notify (cl_device_id Device, cl_event Event, cl_event Finished)
{
  _cl_command_node *Node = Event->command;

  if (Finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (Event);
      return;
    }

  if (!Node->ready)
    return;

  POCL_MSG_PRINT_SYNAPSE ("notify on event %zu \n", Event->id);

  if (pocl_command_is_ready (Node->sync.event.event))
    {
      pocl_update_event_submitted (Event);
      synapse_push_command (Device, Node);
    }

  return;
}

void
pocl_synapse_update_event (cl_device_id Device, cl_event Event)
{
  if (Event->data == NULL && Event->status == CL_QUEUED)
    {
      Event->data = new SynapseCond;
      assert(Event->data);
    }
}

void
pocl_synapse_wait_event (cl_device_id Device, cl_event Event)
{
  POCL_MSG_PRINT_SYNAPSE(" device->wait_event on event %zu\n", Event->id);
  SynapseCond *Cond = (SynapseCond *)Event->data;

  POCL_LOCK_OBJ (Event);
  while (Event->status > CL_COMPLETE) {
    Cond->wait(Event->pocl_lock);
  }
  POCL_UNLOCK_OBJ (Event);
}

void
pocl_synapse_free_event_data (cl_event Event)
{
  assert(Event->data != NULL);
  SynapseCond *Cond = (SynapseCond *)Event->data;
  delete Cond;
  Event->data = nullptr;
}

/****************************************************************************************/

static bool
synapseProcessEvent (SynapseDeviceData *D)
{
  _cl_command_node *Cmd = nullptr;
  bool doExit = D->ExitRequested;

  POCL_LOCK (D->WakeupLock);

  while (doExit == false) {
    Cmd = D->WorkQueue;
    doExit = D->ExitRequested;
    if ((Cmd == nullptr) && (doExit == false)) {
      POCL_WAIT_COND (D->WakeupCond, D->WakeupLock);
    }

    Cmd = D->WorkQueue;
    if (Cmd) {
      DL_DELETE (D->WorkQueue, Cmd);
      POCL_UNLOCK (D->WakeupLock);

      assert (pocl_command_is_ready (Cmd->sync.event.event));
      assert (Cmd->sync.event.event->status == CL_SUBMITTED);

      pocl_exec_command(Cmd);

      POCL_LOCK (D->WakeupLock);
    }

    doExit = D->ExitRequested;
  }

  POCL_UNLOCK (D->WakeupLock);
  return doExit;
}


static void *
synapseDriverThread (void *Dev)
{
  cl_device_id Device = (cl_device_id)Dev;
  SynapseDeviceData *D = (SynapseDeviceData *)Device->data;

  while (1)
    {
      if (synapseProcessEvent (D))
        break;
    }

  POCL_EXIT_THREAD (NULL);
}
