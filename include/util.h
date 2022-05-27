//
// Created by heeve on 27.05.22.
//

#ifndef CUDA_MEMORY_OPTIMIZATIONS_UTIL_H
#define CUDA_MEMORY_OPTIMIZATIONS_UTIL_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <functional>

#define check(ans) { assert_((ans), __FILE__, __LINE__); }

inline void assert_(cudaError_t code, const char *file, int line) {
    if (code == cudaSuccess) return;
    std::cerr << "check failed: " << cudaGetErrorString(code) << " : " << file << '@' << line << std::endl;
    abort();
}

template <typename F>
float cudaEventProfile(const F &func) {
    cudaEvent_t start, stop;

    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start, 0));
    func();
    check(cudaEventRecord(stop, 0));
    check(cudaEventSynchronize(stop));

    float time;
    check(cudaEventElapsedTime(&time, start, stop));

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    return time;
}


#endif //CUDA_MEMORY_OPTIMIZATIONS_UTIL_H
