#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <memory>
#include <limits>

#include "ze_graph_ext.h"

#define GRAPH_EXT_NAME ZE_GRAPH_EXT_NAME_1_5
#define GRAPH_EXT_VERSION ZE_GRAPH_EXT_VERSION_1_5
typedef ze_graph_dditable_ext_1_5_t graph_dditable_ext_t;

using namespace std;

// byte size of raw image data (3x224x224)
#define IMG_LEN 150528

// output classes
#define NUM_CLASSES 1004
#define CLASS_LEN (sizeof(float) * NUM_CLASSES)


int main(int, char **)
{

  ze_result_t Res = zeInit(0);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeINIT failed\n";
    return 1;
  }

  uint32_t DriverCount = 4;
  ze_driver_handle_t DrvHandles[4];
  Res = zeDriverGet(&DriverCount, DrvHandles);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeDriverGet FAILED\n";
    return 1;
  }
  if (DriverCount < 1) {
    std::cerr << "zeDriverGet returned zero drivers\n";
    return 1;
  }
  ze_driver_handle_t DriverH = DrvHandles[0];

  ze_context_desc_t contextDesc = {.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                                   .pNext = nullptr,
                                   .flags = 0};

  ze_context_handle_t ContextH = nullptr;
  Res = zeContextCreate(DriverH, &contextDesc, &ContextH);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeContextCreate FAILED\n";
    return 1;
  }

  uint32_t DeviceCount = 0;
  Res = zeDeviceGet(DriverH, &DeviceCount, nullptr);
  if (Res != ZE_RESULT_SUCCESS || DeviceCount == 0) {
    std::cerr << "zeDeviceGet 1 FAILED\n";
    return 1;
  }

  ze_device_handle_t DeviceH = nullptr;
  DeviceCount = 1;
  Res = zeDeviceGet(DriverH, &DeviceCount, &DeviceH);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeDeviceGet 2 FAILED\n";
    return 1;
  }

  graph_dditable_ext_t *GraphDDITable = nullptr;
  Res = zeDriverGetExtensionFunctionAddress(DriverH, GRAPH_EXT_NAME,
                                 reinterpret_cast<void **>(&GraphDDITable));
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeDriverGetExtensionFunctionAddress FAILED\n";
    return 1;
  }

  ze_command_queue_handle_t QueueH = nullptr;
  ze_command_list_handle_t ListH = nullptr;

    ze_command_queue_desc_t cmdQueueDesc = {.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                            .pNext = nullptr,
                                            .ordinal = 0,
                                            .index = 0,
                                            .flags = 0,
                                            .mode = ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                            .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  Res = zeCommandQueueCreate(ContextH, DeviceH, &cmdQueueDesc, &QueueH);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeCommandQueueCreate FAILED\n";
    return 1;
  }

    ze_command_list_desc_t cmdListDesc = {.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                          .pNext = nullptr,
                                          .commandQueueGroupOrdinal = 0,
                                          .flags = 0};

  Res = zeCommandListCreate(ContextH, DeviceH, &cmdListDesc, &ListH);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeCommandListCreate FAILED\n";
    return 1;
  }


    ze_host_mem_alloc_desc_t hostMemAllocDesc = {.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                                 .pNext = nullptr,
                                                 .flags = 0};

  void* inputBuf = nullptr;
  void* outputBuf = nullptr;
  Res = zeMemAllocHost(ContextH, &hostMemAllocDesc, IMG_LEN, 4096, &inputBuf);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeMemAllocHost inputBuf FAILED\n";
    return 1;
  }
  assert(inputBuf);

  Res = zeMemAllocHost(ContextH, &hostMemAllocDesc, CLASS_LEN, 4096, &outputBuf);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeMemAllocHost outputBuf FAILED\n";
    return 1;
  }
  assert(outputBuf);

  int len = 0;

  std::ifstream InputFile("pixels");
  std::vector<uint8_t> InputData;

  InputFile.seekg(0, InputFile.end);
  len = InputFile.tellg();
  InputFile.seekg(0, InputFile.beg);
  assert(len == IMG_LEN);
  InputData.resize(len);
  InputFile.read((char*)InputData.data(), len);
  std::cerr << "input data len: " << len << "\n";

  std::memcpy(inputBuf, (void*)InputData.data(), IMG_LEN);

  std::ifstream ModelXmlFile("googlenet-v1.xml");
  std::ifstream ModelBinFile("googlenet-v1.bin");

  std::vector<uint8_t> ModelXml;
  std::vector<uint8_t> ModelBin;

  ModelXmlFile.seekg(0, ModelXmlFile.end);
  len = ModelXmlFile.tellg();
  ModelXmlFile.seekg(0, ModelXmlFile.beg);
  ModelXml.resize(len);
  ModelXmlFile.read((char*)ModelXml.data(), len);
  std::cerr << "model.xml len: " << len << "\n";

  ModelBinFile.seekg(0, ModelBinFile.end);
  len = ModelBinFile.tellg();
  ModelBinFile.seekg(0, ModelBinFile.beg);
  ModelBin.resize(len);
  ModelBinFile.read((char*)ModelBin.data(), len);
  std::cerr << "model.bin len: " << len << "\n";

  std::vector<uint8_t> Model;

  const char* BuildFlags = R"RAW(--inputs_precisions="data:I8" --inputs_layouts="data:NCHW"  --outputs_precisions="prob:FP32" --outputs_layouts="prob:NC" --config NPU_PLATFORM="3720" LOG_LEVEL="LOG_DEBUG")RAW";

  // compile XML to BIN
  std::cerr << "Starting Graph compileFromXmlBin\n";

  ze_device_graph_properties_t pDeviceGraphProperties;
  GraphDDITable->pfnDeviceGetGraphProperties(DeviceH, &pDeviceGraphProperties);

  ze_graph_compiler_version_info_t version = {
    .major = pDeviceGraphProperties.compilerVersion.major,
    .minor = pDeviceGraphProperties.compilerVersion.minor};

  uint64_t XmlLen = ModelXml.size();
  assert(XmlLen > 100);
  uint64_t BinLen = ModelBin.size();
  assert(BinLen > 100);

  uint32_t NumInputs = 2;
  uint64_t ModelSize = sizeof(version) + sizeof(NumInputs) + sizeof(XmlLen) + XmlLen +
        sizeof(BinLen) + BinLen;
  std::cerr << "@@@@@ XML LEN " << XmlLen << " BIN LEN " << BinLen << " TOTAL " << ModelSize << "\n";

  Model.resize(ModelSize);

  uint64_t offset = 0;
  memcpy(Model.data(), &version, sizeof(version));
  offset += sizeof(version);

  memcpy(Model.data()+offset, &NumInputs, sizeof(NumInputs));
  offset += sizeof(NumInputs);

  memcpy(Model.data()+offset, &XmlLen, sizeof(XmlLen));
  offset += sizeof(XmlLen);

  memcpy(Model.data()+offset, ModelXml.data(), XmlLen);
  offset += XmlLen;

  memcpy(Model.data()+offset, &BinLen, sizeof(BinLen));
  offset += sizeof(BinLen);

  memcpy(Model.data()+offset, ModelBin.data(), BinLen);
  offset += BinLen;

  assert(offset == ModelSize);

  ze_graph_desc_t GraphDesc = {
    .stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
    .pNext = nullptr,
    .format = ZE_GRAPH_FORMAT_NGRAPH_LITE,
    .inputSize = Model.size(),
    .pInput = Model.data(),
    .pBuildFlags = BuildFlags
  };

  ze_graph_handle_t GraphH = nullptr;
  Res = GraphDDITable->pfnCreate(ContextH, DeviceH,
                                             &GraphDesc, &GraphH);
  bool Success = (Res == ZE_RESULT_SUCCESS);
  if (!Success) {
    std::cerr << "LevelZero: Graph compilation failed with : " << Res << "\n";
  }
  uint32_t logSize = 0;
  Res = GraphDDITable->pfnBuildLogGetString(GraphH, &logSize, nullptr);
  if (Res == ZE_RESULT_SUCCESS && logSize > 0) {
    std::string TempBuildLog;
    TempBuildLog.resize(logSize+1, 0);
    Res = GraphDDITable->pfnBuildLogGetString(GraphH, &logSize,
                                              TempBuildLog.data());
    std::cerr << "buildlog: " << TempBuildLog.c_str() << "\n";
  } else {
    std::cerr << "buildlog empty or failed to get: " << Res << "\n";
  }

  if (!Success) {
    std::cerr << "LevelZero: Graph compilation failed\n";
    return false;
  }

/*
  std::vector<uint8_t> VpuNativeBinary;

  size_t NativeSize = 0;
  Res = GraphDDITable->pfnGetNativeBinary(GraphH, &NativeSize, nullptr);
  if (Res != ZE_RESULT_SUCCESS || NativeSize == 0) {
    std::cerr << "LevelZero: Failed to get Native binary SIZE for Graph\n";
    return false;
  }

  VpuNativeBinary.resize(NativeSize, 0);
  Res = GraphDDITable->pfnGetNativeBinary(GraphH, &NativeSize,
                                          VpuNativeBinary.data());
  if (Res != ZE_RESULT_SUCCESS || NativeSize == 0) {
    std::cerr << "LevelZero: Failed to get Native binary for Graph\n";
    return false;
  }

  // load native graph
  assert(!VpuNativeBinary.empty());
  ze_graph_desc_t GraphDesc2 = {
    .stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
    .pNext = nullptr,
    .format = ZE_GRAPH_FORMAT_NATIVE,
    .inputSize = VpuNativeBinary.size(),
    .pInput = VpuNativeBinary.data(),
    .pBuildFlags = nullptr
  };

  ze_graph_handle_t GraphHFinal = nullptr;
  Res = GraphDDITable->pfnCreate(ContextH, DeviceH,
                                             &GraphDesc2, &GraphHFinal);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "Graph create failed with error: ";
    std::cerr << std::to_string(Res);
    std::cerr << "\n";
    return 1;
  } else {
    assert(GraphHFinal);
    std::cerr << "Graph create SUCCESS\n ";
  }
*/
  ze_graph_handle_t GraphHFinal =  GraphH;

  Res = GraphDDITable->pfnSetArgumentValue(GraphHFinal, 0, inputBuf);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeKernelSetArgumentValue 0 failed\n";
    return 1;
  }

  Res = GraphDDITable->pfnSetArgumentValue(GraphHFinal, 1, outputBuf);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeKernelSetArgumentValue 1 failed\n";
    return 1;
  }


  const uint64_t SyncTimeout = 2'000'000'000;    // 2 seconds
  Success = (zeCommandListReset(ListH) == ZE_RESULT_SUCCESS);
  Res = GraphDDITable->pfnAppendGraphInitialize(ListH, GraphHFinal,
                                                nullptr, 0, nullptr);
  Success = Success && (Res == ZE_RESULT_SUCCESS);
  Success = Success && (zeCommandListClose(ListH) == ZE_RESULT_SUCCESS);
  Success = Success && (zeCommandQueueExecuteCommandLists(QueueH, 1, &ListH, nullptr) == ZE_RESULT_SUCCESS);
  Success = Success && (zeCommandQueueSynchronize(QueueH, SyncTimeout) == ZE_RESULT_SUCCESS);
  if (!Success) {
    std::cerr << "graph initialization via cmdQ failed\n";
    return 1;
  }
  Res = zeCommandListReset(ListH);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeCommandListReset failed\n";
    return 1;
  }

  Res = GraphDDITable->pfnAppendGraphExecute(ListH, GraphHFinal,
                                   nullptr, nullptr, 0, nullptr);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "pfnAppendGraphExecute failed with: " << Res << "\n";
    return 1;
  }

  Res = zeCommandListClose(ListH);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeCommandListClose failed\n";
    return 1;
  }

  Res = zeCommandQueueExecuteCommandLists(QueueH, 1, &ListH, nullptr);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeCommandQueueExecuteCommandLists failed\n";
    return 1;
  }

  Res = zeCommandQueueSynchronize(QueueH, std::numeric_limits<uint64_t>::max());
  if (Res != ZE_RESULT_SUCCESS) {
    std::cerr << "zeCommandQueueSynchronize failed\n";
    return 1;
  }

  std::cerr << "START CLASSIFICATIONS\n";
  float* outputAsFloats = reinterpret_cast<float*>(outputBuf);
  for (unsigned i = 0; i < NUM_CLASSES; ++i) {
    if (i < 4)
       std::cerr << "Output "<< i << ": " << outputAsFloats[i] << "\n";
    if (outputAsFloats[i] > 0.001f) {
       std::cerr << "Classification " << i << " : " << outputAsFloats[i] << "\n";
    }
  }
  std::cerr << "FINISH CLASSIFICATIONS\n";

  return 0;
}
