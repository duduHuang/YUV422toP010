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
		/*
		rgb.x = (p_Src[j * nSrcPitch + k + 0] + p_Src[j * nSrcPitch + k + 3] + p_Src[j * nSrcPitch + k + 6] + p_Src[j * nSrcPitch + k + 9] +
		    p_Src[(j + 1) * nSrcPitch + k + 0] + p_Src[(j + 1) * nSrcPitch + k + 3] + p_Src[(j + 1) * nSrcPitch + k + 6] + p_Src[(j + 1) * nSrcPitch + k + 9] +
			p_Src[(j + 2) * nSrcPitch + k + 0] + p_Src[(j + 2) * nSrcPitch + k + 3] + p_Src[(j + 2) * nSrcPitch + k + 6] + p_Src[(j + 2) * nSrcPitch + k + 9] +
			p_Src[(j + 3) * nSrcPitch + k + 0] + p_Src[(j + 3) * nSrcPitch + k + 3] + p_Src[(j + 3) * nSrcPitch + k + 6] + p_Src[(j + 3) * nSrcPitch + k + 9]) / 16;
		rgb.y = (p_Src[j * nSrcPitch + k + 1] + p_Src[j * nSrcPitch + k + 4] + p_Src[j * nSrcPitch + k + 7] + p_Src[j * nSrcPitch + k + 10] +
		    p_Src[(j + 1) * nSrcPitch + k + 1] + p_Src[(j + 1) * nSrcPitch + k + 4] + p_Src[(j + 1) * nSrcPitch + k + 7] + p_Src[(j + 1) * nSrcPitch + k + 10] +
			p_Src[(j + 2) * nSrcPitch + k + 1] + p_Src[(j + 2) * nSrcPitch + k + 4] + p_Src[(j + 2) * nSrcPitch + k + 7] + p_Src[(j + 2) * nSrcPitch + k + 10] +
			p_Src[(j + 3) * nSrcPitch + k + 1] + p_Src[(j + 3) * nSrcPitch + k + 4] + p_Src[(j + 3) * nSrcPitch + k + 7] + p_Src[(j + 3) * nSrcPitch + k + 10]) / 16;
		rgb.z = (p_Src[j * nSrcPitch + k + 2] + p_Src[j * nSrcPitch + k + 5] + p_Src[j * nSrcPitch + k + 8] + p_Src[j * nSrcPitch + k + 11] +
		    p_Src[(j + 1) * nSrcPitch + k + 2] + p_Src[(j + 1) * nSrcPitch + k + 5] + p_Src[(j + 1) * nSrcPitch + k + 8] + p_Src[(j + 1) * nSrcPitch + k + 11] +
			p_Src[(j + 2) * nSrcPitch + k + 2] + p_Src[(j + 2) * nSrcPitch + k + 5] + p_Src[(j + 2) * nSrcPitch + k + 8] + p_Src[(j + 2) * nSrcPitch + k + 11] +
			p_Src[(j + 3) * nSrcPitch + k + 2] + p_Src[(j + 3) * nSrcPitch + k + 5] + p_Src[(j + 3) * nSrcPitch + k + 8] + p_Src[(j + 3) * nSrcPitch + k + 11]) / 16;
			*/
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
