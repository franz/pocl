#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <iostream>
#include <random>

#include "poclu.h"

#define BUILTIN_KER "pocl.gemm.fp32"

using namespace std;

// matrix W/H
#define MAT_N 64

// byte size of raw matrix
#define MAT_ELEMS (MAT_N * MAT_N)
#define MATRIX_SIZE (MAT_ELEMS * sizeof(float))

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
        return 77;
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

    std::random_device RandomDevice;
    std::mt19937 Mersenne{RandomDevice()};
//    std::uniform_real_distribution<float> UniDist{10.0f, 100.0f};

    float InputA[MAT_ELEMS];
    float InputB[MAT_ELEMS];
    float Output[MAT_ELEMS];
    for (size_t i = 0; i <  MAT_ELEMS ; ++i) {
      Output[i] = (float)11.125;
      InputA[i] = (float)1.25;
      InputB[i] = (float)1.0;
//      InputA[i] = (float)UniDist(Mersenne);
//      InputB[i] = (float)UniDist(Mersenne);
    }

    cl::Buffer Inbuf1{ClContext, (cl_mem_flags)(CL_MEM_READ_ONLY), MATRIX_SIZE};
    cl::Buffer Inbuf2{ClContext, (cl_mem_flags)(CL_MEM_READ_ONLY), MATRIX_SIZE};
    cl::Buffer Outbuf{ClContext, (cl_mem_flags)(CL_MEM_WRITE_ONLY), MATRIX_SIZE};

    NpuQueue.enqueueWriteBuffer(Inbuf1, 0, 0, MATRIX_SIZE, InputA);
    NpuQueue.enqueueWriteBuffer(Inbuf2, 0, 0, MATRIX_SIZE, InputB);
    NpuQueue.enqueueWriteBuffer(Outbuf, 0, 0, MATRIX_SIZE, Output);
    NpuKernel.setArg(0, Inbuf1);
    NpuKernel.setArg(1, Inbuf2);
    NpuKernel.setArg(2, Outbuf);
    NpuQueue.enqueueNDRangeKernel(NpuKernel, cl::NullRange, cl::NDRange(1));
    NpuQueue.enqueueReadBuffer(Outbuf, 0, 0, MATRIX_SIZE, Output);
    NpuQueue.finish();

    for (unsigned i = 0; i < 16; ++i) {
      std::cout << "Matmul[" << i << "] : " << (float)Output[i] << "\n";
    }
  }

  catch (cl::Error& err) {
    std::cout << "FAIL with OpenCL error = " << err.err() << " what: " << err.what() << std::endl;
    return 11;
  }

  return 0;
}
