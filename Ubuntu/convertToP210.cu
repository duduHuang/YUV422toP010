#include <cuda.h>
#include <cuda_runtime.h>

#include "convertToP210.h"

__global__ static void convertToP010Kernel(const uint16_t *pV210, uint16_t *pP010,
    int nDstWidth, int nDstHeight, int nBatch) {
    //int index = threadIdx.x;
    //int stride = blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
    uint4 pF;
    //for (int j = 0; j < (nDstHeight * 4 / 3); j += 1) {
        if (tid < (nDstWidth / 8) && tidd < (nDstHeight * 4 / 3)) {
            int j = tidd;
        //for (int i = index; i < (nDstWidth / 8); i += stride) {
            int k = tid * 8;
            //int k = i * 8;
            pF.x = (uint32_t)pV210[j * nDstWidth + k + 0] + ((uint32_t)pV210[j * nDstWidth + k + 1] << 16);
            pF.y = (uint32_t)pV210[j * nDstWidth + k + 2] + ((uint32_t)pV210[j * nDstWidth + k + 3] << 16);
            pF.z = (uint32_t)pV210[j * nDstWidth + k + 4] + ((uint32_t)pV210[j * nDstWidth + k + 5] << 16);
            pF.w = (uint32_t)pV210[j * nDstWidth + k + 6] + ((uint32_t)pV210[j * nDstWidth + k + 7] << 16);

            v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
            y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
            u0 = (uint32_t)(pF.x & 0x000003FF);
            y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20);
            u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
            y1 = (uint32_t)(pF.y & 0x000003FF);
            u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
            y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10);
            v1 = (uint32_t)(pF.z & 0x000003FF);
            y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20);
            v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
            y4 = (uint32_t)(pF.w & 0x000003FF);

            k = tid * 6;
            //k = i * 6;
            pP010[j * nDstWidth * 3 / 4 + k + 0] = y0;
            pP010[j * nDstWidth * 3 / 4 + k + 1] = y1;
            pP010[j * nDstWidth * 3 / 4 + k + 2] = y2;
            pP010[j * nDstWidth * 3 / 4 + k + 3] = y3;
            pP010[j * nDstWidth * 3 / 4 + k + 4] = y4;
            pP010[j * nDstWidth * 3 / 4 + k + 5] = y5;
            int jj = j * nDstWidth * 3 / 8;
            k = tid * 3;
            //k = i * 3;
            pP010[jj + nDstHeight * nDstWidth + k + 0] = u0;
            pP010[jj + nDstHeight * nDstWidth + k + 1] = u1;
            pP010[jj + nDstHeight * nDstWidth + k + 2] = u2;
            pP010[jj + nDstHeight * nDstWidth * 3 / 2 + k + 0] = v0;
            pP010[jj + nDstHeight * nDstWidth * 3 / 2 + k + 1] = v1;
            pP010[jj + nDstHeight * nDstWidth * 3 / 2 + k + 2] = v2;
        }
    //}
}

void convertToP010(uint16_t *dpSrc, uint16_t * dpDst, int nDstWidth, int nDstHeight,
    int nBatch, cudaStream_t stream) {
    // Restricting blocks in Z-dim till 32 to not launch too many blocks
    dim3 blocks(32, 16, 1);
    dim3 grids((nDstWidth + blocks.x - 1) / blocks.x, (((nDstHeight * 4 / 3) + blocks.y) - 1) / blocks.y, 1);
    convertToP010Kernel << <grids, blocks, 0, stream >> > (dpSrc, dpDst, nDstWidth, nDstHeight, nBatch);
}

__global__ static void convertToP210Kernel(uint16_t *pV210, uint16_t *dP210, int nWidth, int nHeight) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tidd = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
	uint4 pF;
	int nDstW = nWidth / 8;
	int nDstH = nHeight * 4 / 3;
	if (tid < nDstW && tidd < nDstH) {
		int k = tid * 8;
		int j = tidd * nWidth;
		pF.x = (uint32_t)pV210[j + k + 0] + ((uint32_t)pV210[j + k + 1] << 16);
		pF.y = (uint32_t)pV210[j + k + 2] + ((uint32_t)pV210[j + k + 3] << 16);
		pF.z = (uint32_t)pV210[j + k + 4] + ((uint32_t)pV210[j + k + 5] << 16);
		pF.w = (uint32_t)pV210[j + k + 6] + ((uint32_t)pV210[j + k + 7] << 16);

		v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
		y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
		u0 = (uint32_t)(pF.x & 0x000003FF);
		y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20);
		u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
		y1 = (uint32_t)(pF.y & 0x000003FF);
		u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
		y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10);
		v1 = (uint32_t)(pF.z & 0x000003FF);
		y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20);
		v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
		y4 = (uint32_t)(pF.w & 0x000003FF);

		k = tid * 6;
		j = tidd * nWidth * 3 / 4;
		dP210[j + k + 0] = y0;
		dP210[j + k + 1] = y1;
		dP210[j + k + 2] = y2;
		dP210[j + k + 3] = y3;
		dP210[j + k + 4] = y4;
		dP210[j + k + 5] = y5;
		k = tid * 3;
		j = tidd * nWidth * 3 / 8 + nWidth * nHeight;
		dP210[j + k + 0] = u0;
		dP210[j + k + 1] = u1;
		dP210[j + k + 2] = u2;
		j = tidd * nWidth * 3 / 8 + nWidth * nHeight * 3 / 2;
		dP210[j + k + 0] = v0;
		dP210[j + k + 1] = v1;
		dP210[j + k + 2] = v2;
	}
}

void convertToP210(uint16_t *pV210, uint16_t *dP210, int nWidth, int nHeight, cudaStream_t stream) {
	dim3 blocks(32, 16, 1);
	dim3 grids((nWidth + blocks.x - 1) / blocks.x, ((nHeight * 4 / 3) + blocks.y - 1) / blocks.y, 1);
	convertToP210Kernel << < grids, blocks, 0, stream >> > (pV210, dP210, nWidth, nHeight);
}