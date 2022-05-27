#include <stdio.h>
#include "pinning.cuh"

void copyProfiler(float *h_a, float *h_b, float *d, unsigned int n) {
    unsigned int bytes = n * sizeof(float);

    cudaEvent_t start, stop;

    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start, 0));
    check(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    check(cudaEventRecord(stop, 0));
    check(cudaEventSynchronize(stop));

    float time;
    check(cudaEventElapsedTime(&time, start, stop));
    std::cout << "  Host to Device Bandwidth in GB/s: " << bytes * 1e-6 / time << std::endl;

    check(cudaEventRecord(start, 0));
    check(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
    check(cudaEventRecord(stop, 0));
    check(cudaEventSynchronize(stop));

    check(cudaEventElapsedTime(&time, start, stop));
    std::cout << "  Device to Host Bandwidth in GB/s: " << bytes * 1e-6 / time << std::endl;

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
}

void measurePinning(int numElements) {
    const unsigned int bytes = numElements * sizeof(float);
    float *h_aPageable, *h_bPageable, *h_aPinned, *h_bPinned, *d_p;

    std::cout << "Pageable vs Pinned memory transfer comparison\n Transfer size in MB: " << bytes / (1024 * 1024)
              << std::endl;

    // allocate pageable and pinned host memory and device memory
    h_aPageable = (float *) malloc(bytes);
    h_bPageable = (float *) malloc(bytes);
    check(cudaMallocHost((void **) &h_aPinned, bytes));
    check(cudaMallocHost((void **) &h_bPinned, bytes));
    check(cudaMalloc((void **) &d_p, bytes));

    for (int i = 0; i < numElements; ++i) {
        h_aPageable[i] = i;
        h_aPinned[i] = i;
    }

    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    std::cout << "Pageable transfer:" << std::endl;
    copyProfiler(h_aPageable, h_bPageable, d_p, numElements);

    std::cout << "Pinned transfer:" << std::endl;
    copyProfiler(h_aPinned, h_bPinned, d_p, numElements);

    cudaFree(d_p);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
}