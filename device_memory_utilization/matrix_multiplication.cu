//
// Created by heeve on 27.05.22.
//

#include "matrix_multiplication.cuh"
#include "../include/util.h"

template<int TILE_DIM>
__global__ void simpleMultiply(float *a, float *b, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    auto res = 0.0f;
    for (int i = 0; i < TILE_DIM; ++i)
        res += a[row * TILE_DIM + i] * a[col * TILE_DIM + i];

    b[row * M + col] = res;
}

template<int TILE_DIM>
__global__ void coalescedMultiply(float *a, float *b, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float aTile[TILE_DIM][TILE_DIM], aTransposedTile[TILE_DIM][
            TILE_DIM + 1]; // TILE_DIM + 1 - remove bank conflicts

    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    aTransposedTile[threadIdx.x][threadIdx.y] = a[(blockIdx.x * blockDim.x + threadIdx.y) * TILE_DIM + threadIdx.x];
    __syncthreads();

    auto res = 0.0f;
    for (int i = 0; i < TILE_DIM; ++i)
        res += aTile[threadIdx.y][i] * aTransposedTile[i][threadIdx.x];

    b[row * M + col] = res;
}


void coalescedMatrixMultiplication() {
    const int warpNumber = 32;
    const int M = warpNumber * 43;
    const int N = warpNumber * 54;
    size_t numElementsInput = M * N;
    size_t numElementsResult = M * M;
    size_t numBytesInput = sizeof(float) * numElementsInput;
    size_t numBytesResult = sizeof(float) * numElementsResult;

    auto *h_a = (float *) malloc(numBytesInput);
    auto *h_b_simple = (float *) malloc(numBytesResult);
    auto *h_b_coalesced = (float *) malloc(numBytesResult);

    srand(42);
    for (int i = 0; i < numElementsInput; ++i) h_a[i] = rand() / float(RAND_MAX);

    float *d_a, *d_b_simple, *d_b_coalesced;
    check(cudaMalloc((void **) &d_a, numBytesInput));
    check(cudaMalloc((void **) &d_b_simple, numBytesResult));
    check(cudaMalloc((void **) &d_b_coalesced, numBytesResult));

    check(cudaMemcpy(d_a, h_a, numBytesInput, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(warpNumber, warpNumber);
    dim3 blocksPerGrid(N / warpNumber, M / warpNumber);

    // warm-up
    simpleMultiply<warpNumber><<<threadsPerBlock, blocksPerGrid>>>(d_a, d_b_simple, M);
    cudaDeviceSynchronize();

    const int iterations = 500;

    float timeSimpleMultiply = cudaEventProfile([&]() {
        for (int i = 0; i < iterations; ++i)
            simpleMultiply<warpNumber><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b_simple, M);
    });

    cudaDeviceSynchronize();
    timeSimpleMultiply /= iterations;

    float timeCoalescedMultiply = cudaEventProfile([&]() {
        for (int i = 0; i < iterations; ++i)
            coalescedMultiply<warpNumber><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b_coalesced, M);
    });

    cudaDeviceSynchronize();
    timeCoalescedMultiply /= iterations;

    std::cout << "Non-optimized matrix multiplication (C=AA^T) completed in " << timeSimpleMultiply << " ms"
              << std::endl;
    std::cout << "Optimized (smem) matrix multiplication (C=AA^T) completed in " << timeCoalescedMultiply << " ms"
              << std::endl;

    cudaMemcpy(h_b_simple, d_b_simple, numBytesResult, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_coalesced, d_b_coalesced, numBytesResult, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElementsResult; ++i) {
        float simple = h_b_simple[i];
        float coalesced = h_b_coalesced[i];
        if (fabs(simple - coalesced) / warpNumber > 1e-6) {
            std::cout << "results are not the same" << std::endl;
            break;
        }
    }

    free(h_a);
    free(h_b_simple);
    free(h_b_coalesced);
    cudaFree(d_a);
    cudaFree(d_b_simple);
    cudaFree(d_b_coalesced);
}
