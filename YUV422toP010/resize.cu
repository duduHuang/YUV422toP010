#include <cuda.h>
#include <cuda_runtime.h>

#include "resize.h"

__global__ static void resizeBatchKernel(const uint8_t *p_Src, int nSrcPitch, int nSrcHeight, 
	uint8_t *p_dst, int nDstWidth, int nDstHeight, int nBatch) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tidd = blockIdx.y * blockDim.y + threadIdx.y;
	uchar3 rgb;
	if (tid < (nSrcPitch / 12) && tidd < (nSrcHeight * 3 / 4)) {
		int j = tidd * 4;
		int k = tid * 12;
		rgb.x = p_Src[j * nSrcPitch + k + 0];
		rgb.y = p_Src[j * nSrcPitch + k + 1];
		rgb.z = p_Src[j * nSrcPitch + k + 2];
		k = tid * 3;
		j = tidd;
		p_dst[j * nDstWidth + k + 0] = rgb.x;
		p_dst[j * nDstWidth + k + 1] = rgb.y;
		p_dst[j * nDstWidth + k + 2] = rgb.z;
	}
}

void resizeBatch(uint8_t *dpSrc, int nSrcPitch, int nSrcHeight, uint8_t *dpDst, int nDstWidth, int nDstHeight,
	int nBatch, cudaStream_t stram) {
	dim3 blocks(16, 1, 1);
	dim3 grids((nSrcPitch + blocks.x - 1) / blocks.x, (((nSrcHeight * 3) + blocks.y) - 1) / blocks.y, 1);
	resizeBatchKernel << <grids, blocks, 0, stram >> > (dpSrc, nSrcPitch, nSrcHeight, dpDst, nDstWidth, nDstHeight, nBatch);
}
