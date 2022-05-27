#include <iostream>
#include "data_transfers/pinning.cuh"
#include "data_transfers/staged_copy_execute.cuh"
#include "data_transfers/batching.cuh"
#include "device_memory_utilization/matrix_multiplication.cuh"

int main() {
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device used to perform measurements: " << prop.name << std::endl;

    // page-locked memory transfer measurement
//    int numElements = 4 * 1024 * 1024;
//    measurePinning(numElements);
//
//    if (prop.deviceOverlap) // see if concurrent copy & execute is supported by the GPU
//    {
//        stagedCopyExecute(512, 16);
//    }

    coalescedMatrixMultiplication();
    return 0;
}
