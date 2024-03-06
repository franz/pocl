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
#include <random>
#include <synapse_api.h>

#include "synapse_kernel.hh"

static cl_bool DevAvailable = CL_TRUE;
static cl_bool DevUNAvailable = CL_FALSE;

using BIKDMap = std::map<std::string, TensorBIKD *>;

struct SynapseDeviceData {
  std::string Params;
  std::string SupportedList;

  BIKDMap SupportedKernels;

  // wakeup driver thread
  pocl_cond_t WakeupCond;
  pocl_lock_t WakeupLock;

  // List of commands ready to be executed.
  _cl_command_node *WorkQueue = nullptr;

  // driver thread
  pocl_thread_t DriverThread = 0;
  bool ExitRequested = false;

  //synStreamHandle Dev2HostStream = 0, Host2DevStream = 0, ComputeStream = 0;
  synStreamHandle DefaultStream = nullptr;
  synDeviceId DevID = UINT32_MAX;
  synDeviceType DevType = synDeviceTypeInvalid;

  std::random_device RandomDev;
  std::mt19937 GenRand{RandomDev()};
  std::uniform_int_distribution<uint64_t> dist{UINT64_MAX/8, UINT64_MAX};
};

struct SynapseCond {
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
private:
  pocl_cond_t Cond;
};



void pocl_synapse_init_device_ops(struct pocl_device_ops *ops) {

  ops->device_name = "synapse";
  ops->init = pocl_synapse_init;
  ops->uninit = pocl_synapse_uninit;
  ops->probe = pocl_synapse_probe;
  ops->build_hash = pocl_synapse_build_hash;
  ops->setup_metadata = pocl_setup_builtin_metadata;

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
  ops->run = pocl_synapse_run;
  ops->free_event_data = pocl_synapse_free_event_data;
  ops->update_event = pocl_synapse_update_event;
  ops->wait_event = pocl_synapse_wait_event;
  ops->notify_event_finished = pocl_synapse_notify_event_finished;
  ops->notify_cmdq_finished = pocl_synapse_notify_cmdq_finished;
  ops->init_queue = pocl_synapse_init_queue;
  ops->free_queue = pocl_synapse_free_queue;

  ops->build_builtin = pocl_synapse_build_builtin;
  ops->free_program = pocl_synapse_free_program;
  ops->create_kernel = pocl_synapse_create_kernel;
  ops->free_kernel = pocl_synapse_free_kernel;
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

#if 0
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
#endif

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
    for (size_t i = 0; i < PoCL_Tensor_BIDescriptorNum; ++i) {
      if (PoCL_Tensor_BIDescriptors[i].KernelId == KernelId) {
        if (D->SupportedList.size() > 0)
          D->SupportedList += ";";
        D->SupportedList += PoCL_Tensor_BIDescriptors[i].name;
        D->SupportedKernels.insert(std::make_pair(
                                     PoCL_Tensor_BIDescriptors[i].name,
                                     &PoCL_Tensor_BIDescriptors[i]));
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

#if 0
  if (D->ComputeStream) {
    synStreamDestroy(D->ComputeStream);
  }
  if (D->Dev2HostStream) {
    synStreamDestroy(D->Dev2HostStream);
  }
  if (D->Host2DevStream) {
    synStreamDestroy(D->Host2DevStream);
  }
#endif
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
  assert(err == synSuccess);
  err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);
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
  assert(err == synSuccess);
  err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);
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
  assert(err == synSuccess);
  err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);
}

void
pocl_synapse_copy_rect (void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                       cl_mem src_buf,
                       const size_t *__restrict__ const dst_origin,
                       const size_t *__restrict__ const src_origin,
                       const size_t *__restrict__ const region,
                       size_t const dst_row_pitch,
                       size_t const dst_slice_pitch,
                       size_t const src_row_pitch,
                       size_t const src_slice_pitch)
{
  SynapseDeviceData *D = (SynapseDeviceData *)data;
  synStatus err = synSuccess;
  uint64_t src_ptr = (uint64_t)src_mem_id->mem_ptr;
  uint64_t dst_ptr = (uint64_t)dst_mem_id->mem_ptr;
  uint64_t adjusted_src_ptr
      = src_ptr + src_origin[0] + src_row_pitch * src_origin[1]
        + src_slice_pitch * src_origin[2];
  uint64_t adjusted_dst_ptr
      = dst_ptr + dst_origin[0] + dst_row_pitch * dst_origin[1]
        + dst_slice_pitch * dst_origin[2];

  POCL_MSG_PRINT_MEMORY (
      "COPY RECT \n"
      "SRC %zu DST %zu SIZE %zu\n"
      "src origin %u %u %u dst origin %u %u %u \n"
      "src_row_pitch %lu src_slice pitch %lu\n"
      "dst_row_pitch %lu dst_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_src_ptr, adjusted_dst_ptr, region[0] * region[1] * region[2],
      (unsigned)src_origin[0], (unsigned)src_origin[1],
      (unsigned)src_origin[2], (unsigned)dst_origin[0],
      (unsigned)dst_origin[1], (unsigned)dst_origin[2],
      (unsigned long)src_row_pitch, (unsigned long)src_slice_pitch,
      (unsigned long)dst_row_pitch, (unsigned long)dst_slice_pitch,
      (unsigned long)region[0], (unsigned long)region[1],
      (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((src_row_pitch == dst_row_pitch && dst_row_pitch == region[0])
      && (src_slice_pitch == dst_slice_pitch
          && dst_slice_pitch == (region[1] * region[0])))
    {
//      memcpy (adjusted_dst_ptr, adjusted_src_ptr,
//              region[2] * region[1] * region[0]);
      err = synMemCopyAsync(D->DefaultStream,
                      (uint64_t)adjusted_src_ptr,
                      (region[2] * region[1] * region[0]), //size
                      (uint64_t)adjusted_dst_ptr, DRAM_TO_DRAM);
      assert(err == synSuccess);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j) {
//          memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
//                  adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
//                  region[0]);
          err = synMemCopyAsync(D->DefaultStream,
              adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
              region[0], //size
              adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
              DRAM_TO_DRAM);
         assert(err == synSuccess);
       }
    }
  err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);
}

void
pocl_synapse_write_rect (void *data, const void *__restrict__ const host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        const size_t *__restrict__ const buffer_origin,
                        const size_t *__restrict__ const host_origin,
                        const size_t *__restrict__ const region,
                        size_t const buffer_row_pitch,
                        size_t const buffer_slice_pitch,
                        size_t const host_row_pitch,
                        size_t const host_slice_pitch)
{
  SynapseDeviceData *D = (SynapseDeviceData *)data;
  uint64_t device_ptr = (uint64_t)dst_mem_id->mem_ptr;
  synStatus err = synSuccess;
  uint64_t adjusted_device_ptr
      = device_ptr + buffer_origin[0]
        + buffer_row_pitch * buffer_origin[1]
        + buffer_slice_pitch * buffer_origin[2];
  uint64_t adjusted_host_ptr
      = (uint64_t)host_ptr + host_origin[0]
        + host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];

  POCL_MSG_PRINT_MEMORY (
      "WRITE RECT \n"
      "SRC HOST %zu DST DEV %zu SIZE %zu\n"
      "borigin %u %u %u horigin %u %u %u \n"
      "row_pitch %lu slice pitch \n"
      "%lu host_row_pitch %lu host_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_host_ptr, adjusted_device_ptr,
      region[0] * region[1] * region[2], (unsigned)buffer_origin[0],
      (unsigned)buffer_origin[1], (unsigned)buffer_origin[2],
      (unsigned)host_origin[0], (unsigned)host_origin[1],
      (unsigned)host_origin[2], (unsigned long)buffer_row_pitch,
      (unsigned long)buffer_slice_pitch, (unsigned long)host_row_pitch,
      (unsigned long)host_slice_pitch, (unsigned long)region[0],
      (unsigned long)region[1], (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((buffer_row_pitch == host_row_pitch && host_row_pitch == region[0])
      && (buffer_slice_pitch == host_slice_pitch
          && host_slice_pitch == (region[1] * region[0])))
    {
//      memcpy (adjusted_device_ptr, adjusted_host_ptr,
//              region[2] * region[1] * region[0]);
      err = synMemCopyAsync(D->DefaultStream,
                                      adjusted_host_ptr,
                                      (region[2] * region[1] * region[0]), //size
                                      adjusted_device_ptr,
                                      HOST_TO_DRAM);
      assert(err == synSuccess);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j) {
//          memcpy (adjusted_device_ptr + buffer_row_pitch * j
//                      + buffer_slice_pitch * k,
//                  adjusted_host_ptr + host_row_pitch * j
//                      + host_slice_pitch * k,
//                  region[0]);

          err = synMemCopyAsync(D->DefaultStream,
              adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              region[0], //size
              adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              HOST_TO_DRAM);
          assert(err == synSuccess);
        }
    }
  err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);
}

void
pocl_synapse_read_rect (void *data, void *__restrict__ const host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                       const size_t *__restrict__ const buffer_origin,
                       const size_t *__restrict__ const host_origin,
                       const size_t *__restrict__ const region,
                       size_t const buffer_row_pitch,
                       size_t const buffer_slice_pitch,
                       size_t const host_row_pitch,
                       size_t const host_slice_pitch)
{
  SynapseDeviceData *D = (SynapseDeviceData *)data;
  uint64_t device_ptr = (uint64_t)src_mem_id->mem_ptr;
  synStatus err = synSuccess;
  uint64_t adjusted_device_ptr
      = device_ptr + buffer_origin[2] * buffer_slice_pitch
        + buffer_origin[1] * buffer_row_pitch + buffer_origin[0];
  uint64_t adjusted_host_ptr
      = (uint64_t)host_ptr + host_origin[2] * host_slice_pitch
        + host_origin[1] * host_row_pitch + host_origin[0];

  POCL_MSG_PRINT_MEMORY (
      "READ RECT \n"
      "SRC DEV %zu DST HOST %zu SIZE %zu\n"
      "borigin %u %u %u horigin %u %u %u row_pitch %lu slice pitch "
      "%lu host_row_pitch %lu host_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_device_ptr, adjusted_host_ptr,
      region[0] * region[1] * region[2], (unsigned)buffer_origin[0],
      (unsigned)buffer_origin[1], (unsigned)buffer_origin[2],
      (unsigned)host_origin[0], (unsigned)host_origin[1],
      (unsigned)host_origin[2], (unsigned long)buffer_row_pitch,
      (unsigned long)buffer_slice_pitch, (unsigned long)host_row_pitch,
      (unsigned long)host_slice_pitch, (unsigned long)region[0],
      (unsigned long)region[1], (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((buffer_row_pitch == host_row_pitch && host_row_pitch == region[0])
      && (buffer_slice_pitch == host_slice_pitch
          && host_slice_pitch == (region[1] * region[0])))
    {
//      memcpy (adjusted_host_ptr, adjusted_device_ptr,
//              region[2] * region[1] * region[0]);
      err = synMemCopyAsync(D->DefaultStream,
          adjusted_device_ptr,
          (region[2] * region[1] * region[0]), // size
          adjusted_host_ptr,
          DRAM_TO_HOST);
      assert(err == synSuccess);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j) {
//          memcpy (adjusted_host_ptr + host_row_pitch * j
//                      + host_slice_pitch * k,
//                  adjusted_device_ptr + buffer_row_pitch * j
//                      + buffer_slice_pitch * k,
//                  region[0]);
          err = synMemCopyAsync(D->DefaultStream,
              adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              (region[2] * region[1] * region[0]), // size
              adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              DRAM_TO_HOST);
          assert(err == synSuccess);
       }
    }
  err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);
}

void pocl_synapse_memfill(void *data, pocl_mem_identifier *dst_mem_id,
                      cl_mem dst_buf, size_t size, size_t offset,
                      const void *__restrict__ pattern, size_t pattern_size) {
  if (pattern_size > 4) {
    POCL_ABORT_UNIMPLEMENTED("synapse memfill with pattern size > 4");
  }

  SynapseDeviceData *D = (SynapseDeviceData *)data;
  synStatus err = synSuccess;
  char* Dst = (char*)dst_mem_id->mem_ptr + offset;
  uint8_t Pat8 = ((uint8_t*)pattern)[0];
  uint16_t Pat16 = ((uint16_t*)pattern)[0];
  uint32_t Pat32 = ((uint32_t*)pattern)[0];

  switch (pattern_size) {
    case 1:  err = synMemsetD8Async((uint64_t)Dst, Pat8, size, D->DefaultStream); assert(err == synSuccess); break;
    case 2:  err = synMemsetD16Async((uint64_t)Dst, Pat16, size/2, D->DefaultStream); assert(err == synSuccess); break;
    case 4:  err = synMemsetD32Async((uint64_t)Dst, Pat32, size/4, D->DefaultStream); assert(err == synSuccess); break;
    default: POCL_MSG_ERR("unknown pattern size\n");
  }
  err = synDeviceSynchronize(D->DevID);
  assert(err == synSuccess);
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

void pocl_synapse_run(void *data, _cl_command_node *cmd) {
  SynapseDeviceData *D = (SynapseDeviceData *)data;

  cl_kernel Kernel = cmd->command.run.kernel;
  struct pocl_context *PoclContext = &cmd->command.run.pc;

  SynapseKernel *K = (SynapseKernel *)Kernel->data[cmd->program_device_i];
  assert(K);
  K->launch(D->DefaultStream, D->DevID, Kernel, PoclContext, cmd);
}

int pocl_synapse_build_builtin(cl_program program, cl_uint device_i) {

  cl_device_id Dev = program->devices[device_i];
  SynapseDeviceData *D = (SynapseDeviceData *)Dev->data;

  BIKDMap *ProgramKernels = new BIKDMap;
  assert(ProgramKernels);
  program->data[device_i] = (void*)ProgramKernels;

  for (unsigned i = 0; i < program->num_builtin_kernels; ++i) {
    std::string KName(program->builtin_kernel_names[i]);
    if (D->SupportedKernels.find(KName) != D->SupportedKernels.end()) {
      ProgramKernels->emplace(KName, D->SupportedKernels[KName]);
      program->builtin_kernel_descriptors[i] = D->SupportedKernels[KName];
    } else {
      POCL_MSG_ERR("Unknown builtin kernel %s\n", KName.c_str());
      return CL_BUILD_PROGRAM_FAILURE;
    }
  }
  return CL_BUILD_SUCCESS;
}

int pocl_synapse_free_program(cl_device_id device, cl_program program,
                              unsigned program_device_i) {
  BIKDMap *ProgramKernels =
      (BIKDMap *)program->data[program_device_i];

  if (ProgramKernels)
    delete ProgramKernels;

  program->data[program_device_i] = nullptr;
  return CL_SUCCESS;
}


int pocl_synapse_create_kernel (cl_device_id device, cl_program program,
                                cl_kernel kernel, unsigned program_device_i) {
  cl_device_id Dev = program->devices[program_device_i];
  SynapseDeviceData *D = (SynapseDeviceData *)Dev->data;

  BIKDMap *ProgramKernels =
      (BIKDMap *)program->data[program_device_i];

  TensorBIKD *BIKernel = ProgramKernels->at(kernel->name);

  SynapseKernel *K = new SynapseKernel(D->DevID, D->DevType, BIKernel);
  if (!K->init()) {
    POCL_MSG_ERR("Synapse: kernel initialization failed!\n");
    delete K;
    return CL_BUILD_PROGRAM_FAILURE;
  }
  kernel->data[program_device_i] = (void*)K;
  return CL_SUCCESS;
}

int pocl_synapse_free_kernel (cl_device_id device, cl_program program,
                              cl_kernel kernel, unsigned program_device_i) {

  SynapseKernel *K = (SynapseKernel *)kernel->data[program_device_i];
  if (K)
    delete K;
  kernel->data[program_device_i] = nullptr;
  return CL_SUCCESS;
}

//*************************************************************************
//*************************************************************************
//*************************************************************************
//*************************************************************************

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
  assert(Queue->data == nullptr);
  SynapseDeviceData *D = (SynapseDeviceData *)Dev->data;
  SynapseCond *CQ = new SynapseCond;
  if (CQ == nullptr)
    return CL_OUT_OF_HOST_MEMORY;

  Queue->data = CQ;
  return CL_SUCCESS;
}

int
pocl_synapse_free_queue (cl_device_id Dev, cl_command_queue Queue)
{
  SynapseCond *CQ = (SynapseCond *)Queue->data;
  if (CQ)
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
  SynapseCond *CQ = (SynapseCond *)Queue->data;
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
  SynapseCond *CQ = (SynapseCond *)Queue->data;
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

  POCL_LOCK_OBJ(Event);
  while (Event->status > CL_COMPLETE) {
    Cond->wait(Event->pocl_lock);
  }
  POCL_UNLOCK_OBJ(Event);
}

void
pocl_synapse_free_event_data (cl_event Event)
{
  SynapseCond *Cond = (SynapseCond *)Event->data;
  if (Cond)
    delete Cond;
  Event->data = nullptr;
}

/****************************************************************************************/


static void *
synapseDriverThread (void *Dev)
{
  cl_device_id Device = (cl_device_id)Dev;
  SynapseDeviceData *D = (SynapseDeviceData *)Device->data;

  _cl_command_node *Cmd = nullptr;
  bool doExit = D->ExitRequested;

  POCL_LOCK (D->WakeupLock);

  do {
    Cmd = D->WorkQueue;
    doExit = D->ExitRequested;

    if (Cmd) {
      DL_DELETE (D->WorkQueue, Cmd);
      POCL_UNLOCK (D->WakeupLock);

      assert (pocl_command_is_ready (Cmd->sync.event.event));
      assert (Cmd->sync.event.event->status == CL_SUBMITTED);

      pocl_exec_command(Cmd);

      POCL_LOCK (D->WakeupLock);
    } else {
      if (doExit == false) {
        POCL_WAIT_COND (D->WakeupCond, D->WakeupLock);
      }
    }

    doExit = D->ExitRequested;
  } while (doExit == false);

  POCL_UNLOCK (D->WakeupLock);

  POCL_EXIT_THREAD(NULL);
}
