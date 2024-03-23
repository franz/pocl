#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <iostream>

#include "poclu.h"

#define BUILTIN_KER "pocl.googlenet.v1.fp32"

using namespace std;

// byte size of raw image data (3x224x224)
#define IMG_LEN 150528
// classification lenght
#define NUM_CLASSES 1001
#define CLASS_LEN (sizeof(float) * NUM_CLASSES)


int main(int, char **)
{
  try {
    cl::Platform platform;
    std::vector<cl::Device> devices;

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(!all_platforms.size()) {
        std::cerr << "No OpenCL platforms available!\n";
        return 1;
    }
    platform = all_platforms[0];

    cl::Device NpuDev;

    platform.getDevices(CL_DEVICE_TYPE_CUSTOM, &devices);
    if(devices.size() == 0) {
        std::cerr << "No OpenCL 'custom' devices available!\n";
        return 2;
    }
    NpuDev = devices[0];
    std::vector<cl::Device> NpuDevs = {NpuDev};
    cl::Context ClContext{NpuDevs};
    cl::CommandQueue NpuQueue{ClContext};

    std::string NpuBuiltinKernels = NpuDev.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
    const std::string BuiltinKernelName{BUILTIN_KER};
    cl::Program NpuProgram{ClContext, NpuDevs, BuiltinKernelName};
    NpuProgram.build(NpuDevs);
    cl::Kernel NpuKernel{NpuProgram, BuiltinKernelName.c_str()};

    // *****************************************************************

    std::string kernel_name = NpuKernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
    std::string a = NpuKernel.getInfo<CL_KERNEL_ATTRIBUTES>();
    unsigned num_args = NpuKernel.getInfo<CL_KERNEL_NUM_ARGS>();

    for (cl_uint arg_index = 0; arg_index < num_args; ++arg_index) {

        cl_kernel_arg_access_qualifier acc_q = NpuKernel.getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index);
        cl_kernel_arg_address_qualifier addr_q = NpuKernel.getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index);
        cl_kernel_arg_type_qualifier type_q = NpuKernel.getArgInfo<CL_KERNEL_ARG_TYPE_QUALIFIER>(arg_index);

        std::string arg_typename = NpuKernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index);
        std::string arg_name = NpuKernel.getArgInfo<CL_KERNEL_ARG_NAME>(arg_index);

        std::cerr << "KERNEL " << kernel_name << " | ARG " << arg_index << " | NAME " << arg_name << " | TYPE "
                  << arg_typename << " | ACC Q " << acc_q << " | ADDR Q "
                  << addr_q << " | TYPE Q " << type_q << "\n";
    }

    size_t Len = 0;
    char* Content = poclu_read_binfile(SRCDIR "/pixels", &Len);
    float* Classification = new float[NUM_CLASSES];
    TEST_ASSERT(Content != nullptr);
    TEST_ASSERT(Len == IMG_LEN);

    cl::Buffer Inbuf{ClContext, (cl_mem_flags)(CL_MEM_READ_ONLY), IMG_LEN};
    cl::Buffer Outbuf{ClContext, (cl_mem_flags)(CL_MEM_WRITE_ONLY), CLASS_LEN};

    NpuQueue.enqueueWriteBuffer(Inbuf, 0, 0, IMG_LEN, Content);
    NpuKernel.setArg(0, Inbuf);
    NpuKernel.setArg(1, Outbuf);
    NpuQueue.enqueueNDRangeKernel(NpuKernel, cl::NullRange, cl::NDRange(1));
    NpuQueue.enqueueReadBuffer(Outbuf, 0, 0, CLASS_LEN, Classification);
    NpuQueue.finish();

    std::cout << "START CLASSIFICATIONS\n";
    for (unsigned i = 0; i < NUM_CLASSES; ++i) {
      if (Classification[i] > 0.001f) {
         std::cout << "Classification " << i << " : " << Classification[i] << "\n";
      }
    }
    std::cout << "FINISH CLASSIFICATIONS\n";
  }

  catch (cl::Error& err) {
    std::cout << "FAIL with OpenCL error = " << err.err() << " what: " << err.what() << std::endl;
    return 11;
  }

  return 0;
}
