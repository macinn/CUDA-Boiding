#include <iostream>
#include <cuda_runtime.h>

int checkCuda() {
    // Check if CUDA is available
    cudaError_t cudaStatus = cudaRuntimeGetVersion(nullptr);

    if (cudaStatus == cudaSuccess) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount > 0) {
            std::cout << "CUDA is available on this system." << std::endl;
            std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);

            std::cout << "CUDA device name: " << deviceProp.name << std::endl;
        }
        else {
            std::cout << "No CUDA devices found on this system." << std::endl;
        }
    }
    else {
        std::cerr << "CUDA is not available on this system." << std::endl;
    }

    return 0;
}
