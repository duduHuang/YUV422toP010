#include <cuda.h>
#include <cuda_runtime.h>

#include "resize.h"

__global__ static void resizeBatchKernel(const uint8_t *p_Src, int nSrcPitch, int nSrcHeight, 
    uint8_t *p_dst, int nDstWidth, int nDstHeight, int nBatch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uchar3 rgb;
    int yScale = nSrcHeight / nDstHeight;
    int xScale = 3 * (nSrcPitch / nDstWidth);
    if (tid < (nDstWidth / 3) && tidd < (nDstHeight * 3)) {
        int j = tidd * yScale * nSrcPitch;
        int k = tid * xScale;
        rgb.x = p_Src[j + k + 0];
        rgb.y = p_Src[j + k + 1];
        rgb.z = p_Src[j + k + 2];
        k = tid * 3;
        j = tidd * nDstWidth;
        p_dst[j + k + 0] = rgb.x;
        p_dst[j + k + 1] = rgb.y;
        p_dst[j + k + 2] = rgb.z;
    }
}

void resizeBatch(uint8_t *dpSrc, int nSrcPitch, int nSrcHeight, uint8_t *dpDst, int nDstWidth, int nDstHeight,
    int nBatch, cudaStream_t stram) {
    dim3 blocks(32, 32, 1);
    dim3 grids((nSrcPitch + blocks.x - 1) / blocks.x, (((nSrcHeight * 3) + blocks.y) - 1) / blocks.y, 1);
    resizeBatchKernel << <grids, blocks, 0, stram >> > (dpSrc, nSrcPitch, nSrcHeight, dpDst, nDstWidth, nDstHeight, nBatch);
}

__global__ static void resizeBatchKernel(const uint16_t *p_Src, int nSrcPitch, int nSrcHeight,
    uint16_t *p_dst, int nDstWidth, int nDstHeight, int nBatch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
    uint4 pF0, pF1, pF2, pF3;
	int xScale = 8 * (nSrcPitch / nDstWidth);
	int yScale = nSrcHeight / nDstHeight;
    if (tid < (nDstWidth / 8) && tidd < (nDstHeight * 4 / 3)) {
        int j = tidd * nSrcPitch * yScale;
        int k = tid * xScale;

        pF0.x = (uint32_t)p_Src[j + k + 0] + ((uint32_t)p_Src[j + k + 1] << 16);
        pF0.w = (uint32_t)p_Src[j + k + 6];

        pF1.y = (uint32_t)p_Src[j + k + 10] + ((uint32_t)p_Src[j + k + 11] << 16);
        pF1.z = (uint32_t)p_Src[j + k + 12];

        pF2.x = (uint32_t)p_Src[j + k + 16] + ((uint32_t)p_Src[j + k + 17] << 16);
        pF2.z = ((uint32_t)p_Src[j + k + 21] << 16);
        pF2.w = (uint32_t)p_Src[j + k + 22] + ((uint32_t)p_Src[j + k + 23] << 16);

        pF3.y = ((uint32_t)p_Src[j + k + 27] << 16);

        v0 = (uint32_t)((pF0.x & 0x3FF00000) >> 20);
        y0 = (uint32_t)((pF0.x & 0x000FFC00) >> 10);
        u0 = (uint32_t)(pF0.x & 0x000003FF);
        y1 = (uint32_t)(pF0.w & 0x000003FF);

        y2 = (uint32_t)((pF1.y & 0x3FF00000) >> 20);
        u1 = (uint32_t)((pF1.y & 0x000FFC00) >> 10);
        v1 = (uint32_t)(pF1.z & 0x000003FF);

        y3 = (uint32_t)((pF2.x & 0x000FFC00) >> 10);
        u2 = (uint32_t)((pF2.z & 0x3FF00000) >> 20);
        v2 = (uint32_t)((pF2.w & 0x000FFC00) >> 10);
        y4 = (uint32_t)(pF2.w & 0x000003FF);

        y5 = (uint32_t)((pF3.y & 0x3FF00000) >> 20);

        k = tid * 12;
        j = tidd * nDstWidth * 3 / 2;
        p_dst[j + k + 0] = y0;
        p_dst[j + k + 1] = u0;
        p_dst[j + k + 2] = v0;
        p_dst[j + k + 3] = y1;
        p_dst[j + k + 4] = y2;
        p_dst[j + k + 5] = u1;
        p_dst[j + k + 6] = v1;
        p_dst[j + k + 7] = y3;
        p_dst[j + k + 8] = y4;
        p_dst[j + k + 9] = u2;
        p_dst[j + k + 10] = v2;
        p_dst[j + k + 11] = y5;
    }
}

void resizeBatch(uint16_t *dpSrc, int nSrcPitch, int nSrcHeight, uint16_t *dpDst, int nDstWidth, int nDstHeight,
    int nBatch, cudaStream_t stram) {
    dim3 blocks(32, 32, 1);
    dim3 grids((nSrcPitch + blocks.x - 1) / blocks.x, (((nSrcHeight * 4 / 3) + blocks.y) - 1) / blocks.y, 1);
    resizeBatchKernel << <grids, blocks, 0, stram >> > (dpSrc, nSrcPitch, nSrcHeight, dpDst, nDstWidth, nDstHeight, nBatch);
}
