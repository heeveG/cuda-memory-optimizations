//
// Created by heeve on 27.05.22.
//

#include "batching.cuh"
#include <functional>

void copyProfiler(const std::function<void()> &copyLambdaH2D, const std::function<void()> &copyLambdaD2H) {
    cudaEvent_t start, stop;

    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start, 0));
    copyLambdaH2D();
    check(cudaEventRecord(stop, 0));
    check(cudaEventSynchronize(stop));

    float time;
    check(cudaEventElapsedTime(&time, start, stop));
    std::cout << "  Host to Device Time to copy: " << time << std::endl;

    check(cudaEventRecord(start, 0));
    copyLambdaD2H();
    check(cudaEventRecord(stop, 0));
    check(cudaEventSynchronize(stop));

    check(cudaEventElapsedTime(&time, start, stop));
    std::cout << "  Device to Host Time to copy: " << time << std::endl;

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
}

void measureBatching() {
    int numElements = 1024 * 1024;
    const unsigned int bytes = numElements * sizeof(float);
    float *h_a, *h_b, *h_c, *h_d, *h_batched, *d_a, *d_b, *d_c, *d_d, *d_batched;

    h_a = (float *) malloc(bytes);
    h_b = (float *) malloc(bytes);
    h_c = (float *) malloc(bytes);
    h_d = (float *) malloc(bytes);
    check(cudaMalloc((void **) &d_a, bytes));
    check(cudaMalloc((void **) &d_b, bytes));
    check(cudaMalloc((void **) &d_c, bytes));
    check(cudaMalloc((void **) &d_d, bytes));


    for (int i = 0; i < numElements; ++i) h_a[i] = i;
    memcpy(h_b, h_a, bytes);
    memcpy(h_c, h_a, bytes);
    memcpy(h_d, h_a, bytes);

    // measure without batching
    copyProfiler([&]() {
                     check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
                     check(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
                     check(cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice));
                     check(cudaMemcpy(d_d, h_d, bytes, cudaMemcpyHostToDevice));
                 },
                 [&]() {
                     check(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
                     check(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));
                     check(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
                     check(cudaMemcpy(h_d, d_d, bytes, cudaMemcpyDeviceToHost));
                 }
    );

    check(cudaMallocHost((void **) &h_batched, 4 * bytes));
    check(cudaMalloc((void **) &d_batched, 4 * bytes));

    memcpy(h_batched + 0 * numElements, h_a, bytes);
    memcpy(h_batched + 1 * numElements, h_b, bytes);
    memcpy(h_batched + 2 * numElements, h_c, bytes);
    memcpy(h_batched + 3 * numElements, h_d, bytes);

    // measure with batching
    copyProfiler([&]() {
                     check(cudaMemcpy(d_batched, h_batched, 4 * bytes, cudaMemcpyHostToDevice));
                 },
                 [&]() {
                     check(cudaMemcpy(h_batched, d_batched, 4 * bytes, cudaMemcpyDeviceToHost));
                 }
    );

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    cudaFreeHost(h_batched);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_batched);
}