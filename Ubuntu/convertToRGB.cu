#include <cuda.h>
#include <cuda_runtime.h>

#include "convertToRGB.h"

__global__ static void convertToRGBKernel(const uint16_t *pV210, uint16_t *tt,
	int nDstWidth, int nDstHeight, int nBatch) {
	//int index = threadIdx.x;
	//int stride = blockDim.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tidd = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
	uint4 pF;
	//for (int j = 0; j < (nDstHeight * 4 / 3); j += 1) {
	    if (tid < (nDstWidth / 8) && tidd < (nDstHeight * 4 / 3)) {
	    //if (tid < (nDstWidth / 8)) {
			int j = tidd;
			//for (int i = index; i < (nDstWidth / 8); i += stride) {
			int k = tid * 8;
			//int k = i * 8;
			pF.x = (uint32_t)pV210[j * nDstWidth + k + 0] + ((uint32_t)pV210[j * nDstWidth + k + 1] << 16);

			pF.y = (uint32_t)pV210[j * nDstWidth + k + 2] + ((uint32_t)pV210[j * nDstWidth + k + 3] << 16);

			pF.z = (uint32_t)pV210[j * nDstWidth + k + 4] + ((uint32_t)pV210[j * nDstWidth + k + 5] << 16);

			pF.w = (uint32_t)pV210[j * nDstWidth + k + 6] + ((uint32_t)pV210[j * nDstWidth + k + 7] << 16);

			v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
			y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10) * 1164;
			u0 = (uint32_t)(pF.x & 0x000003FF);
			y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20) * 1164;
			u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
			y1 = (uint32_t)(pF.y & 0x000003FF) * 1164;
			u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
			y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10) * 1164;
			v1 = (uint32_t)(pF.z & 0x000003FF);
			y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20) * 1164;
			v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
			y4 = (uint32_t)(pF.w & 0x000003FF) * 1164;

			k = tid * 18;
			//k = i * 18;
			int r = 1596 * v0 - 891648, g = 813 * v0 + 392 * u0 - 542464, b = 2017 * u0 - 1107200;
			tt[j * nDstWidth * 9 / 4 + k + 0] = (y0 + r) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 1] = (y0 - g) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 2] = (y0 + b) / 1000;

			tt[j * nDstWidth * 9 / 4 + k + 3] = (y1 + r) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 4] = (y1 - g) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 5] = (y1 + b) / 1000;

			r = 1596 * v1 - 891648, g = 813 * v1 + 392 * u1 - 542464, b = 2017 * u1 - 1107200;
			tt[j * nDstWidth * 9 / 4 + k + 6] = (y2 + r) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 7] = (y2 - g) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 8] = (y2 + b) / 1000;

			tt[j * nDstWidth * 9 / 4 + k + 9] = (y3 + r) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 10] = (y3 - g) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 11] = (y3 + b) / 1000;

			r = 1596 * v2 - 891648, g = 813 * v2 + 392 * u2 - 542464, b = 2017 * u2 - 1107200;
			tt[j * nDstWidth * 9 / 4 + k + 12] = (y4 + r) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 13] = (y4 - g) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 14] = (y4 + b) / 1000;

			tt[j * nDstWidth * 9 / 4 + k + 15] = (y5 + r) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 16] = (y5 - g) / 1000;
			tt[j * nDstWidth * 9 / 4 + k + 17] = (y5 + b) / 1000;
		}
	//}
}

void convertToRGB(uint16_t *dpSrc, int nSrcPitch, uint16_t *dpDst, int nDstWidth, int nDstHeight,
	int nBatch, int block_size, cudaStream_t stram) {
	dim3 blocks(16, 1, 1);
	dim3 grids((7680 + blocks.x - 1) / blocks.x, (((4320 * 4 / 3) + blocks.y - 1)) / blocks.y, 1);
	//int blocks = 32;
	//int grids = (nSrcPitch + blocks / blocks);
	convertToRGBKernel << <grids, blocks, 0, stram >> > (dpSrc, dpDst, nSrcPitch, nDstHeight, nBatch);
}
