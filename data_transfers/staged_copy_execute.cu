//
// Created by heeve on 27.05.22.
//

#include "staged_copy_execute.cuh"
#include "../include/util.h"

__global__ void kernel(float *a, int streamOffset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + streamOffset;

    auto iFloat = (float) i;
    float sin = sinf(iFloat);
    float cos = cosf(iFloat);

    a[i] += sqrtf(sin * sin + cos * cos);
}

void stagedCopyExecute(int blockSize, int numStreams) {

    int numElements = blockSize * numStreams * 8 * 1024;
    int bytes = numElements * sizeof(float);

    int streamSize = numElements / numStreams;
    int streamBytes = streamSize * sizeof(float);

    float *h_a, *d_a;

    check(cudaMallocHost((void **) &h_a, bytes));
    check(cudaMalloc((void **) &d_a, bytes));

    cudaStream_t streams[numStreams];

    for (auto &stream : streams) check(cudaStreamCreate(&stream));

    // sequential version
    memset(h_a, 0, bytes);
    float timeSeq = cudaEventProfile([&]() {
        check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        kernel<<<numElements / blockSize, blockSize>>>(d_a, 0);
        check(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    });

    std::cout <<"Time for sequential copy-execute in ms " <<  timeSeq << std::endl;

    // async version
    memset(h_a, 0, bytes);
    float timeAsync = cudaEventProfile([&]() {
        for (int i = 0; i < numStreams; ++i) {
            int offset = i * streamSize;
            check(cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes, cudaMemcpyHostToDevice,
                                  streams[i]));
        }
        for (int i = 0; i < numStreams; ++i) {
            int offset = i * streamSize;
            kernel<<<streamSize / blockSize, blockSize, 0, streams[i]>>>(d_a, offset);
        }
        for (int i = 0; i < numStreams; ++i) {
            int offset = i * streamSize;
            check(cudaMemcpyAsync(&h_a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost,
                                  streams[i]));
        }
    });

    std::cout << "Time for asynchronous copy-execute in ms: " << timeAsync << std::endl;

    for (auto& stream : streams) check(cudaStreamDestroy(stream));

    cudaFree(d_a);
    cudaFreeHost(h_a);
}