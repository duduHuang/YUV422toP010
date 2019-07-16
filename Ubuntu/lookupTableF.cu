#include <cuda.h>
#include <cuda_runtime.h>

#include "lookupTableF.h"

__global__ static void lookupTableFKernel(int *lookupTable) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 1024) {
        lookupTable[tid] = round(tid * 0.249);
    }
}


void lookupTableF(int *lookupTable, cudaStream_t stream) {
    int blocks = 256;
    int grids = ((1024 + blocks) / blocks);
    lookupTableFKernel << <grids, blocks, 0, stream >> > (lookupTable);
}