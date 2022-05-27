//
// Created by heeve on 27.05.22.
//

#ifndef CUDA_MEMORY_OPTIMIZATIONS_STAGED_COPY_EXECUTE_CUH
#define CUDA_MEMORY_OPTIMIZATIONS_STAGED_COPY_EXECUTE_CUH

#include "../include/util.h"

void stagedCopyExecute(int blockSize, int numStreams);

#endif //CUDA_MEMORY_OPTIMIZATIONS_STAGED_COPY_EXECUTE_CUH
