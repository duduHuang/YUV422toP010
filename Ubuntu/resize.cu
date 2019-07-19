#include <cuda.h>
#include <cuda_runtime.h>

#include "resize.h"

__global__ static void resizeBatchKernel(const uint8_t *p_Src, int nSrcPitch, int nSrcHeight, 
    uint8_t *p_dst, int nDstWidth, int nDstHeight, int nBatch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tidd = blockIdx.y * blockDim.y + threadIdx.y;
    uchar3 rgb;
    int nDstW = nDstWidth / 3;
    int nDstH = nDstHeight * 3;
    int yScale = nSrcHeight / nDstHeight;
    int xScale = 3 * (nSrcPitch / nDstWidth);
    if (tid < nDstW && tidd < nDstH) {
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

    if (tid < nDstHeight * nDstWidth * 2)
        p_dst[tid] = 1023;
    uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
    uint4 pF;
    int xScale = 4;
    int yScale = 4;
    int nDstH = nDstHeight;
    int nDstW = nDstWidth / 6;
    //int nDstH = nDstHeight * 2;
    //int nDstW = nDstWidth;
    if (tid < nDstW && tidd < nDstH) {

       /* int j = tidd * nSrcPitch * yScale;
        int k = tid * xScale;
        y0 = p_Src[j + k + 0];
        k = tid;
        j = tidd * nDstWidth;
        p_dst[j + k + 0] = y0;*/


        int j = tidd * nSrcPitch * 4 / 3 * yScale;
        int k = tid * 32;

        pF.x = (uint32_t)p_Src[j + k + 0] + ((uint32_t)p_Src[j + k + 1] << 16);
        pF.w = (uint32_t)p_Src[j + k + 6];

        v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
        y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
        u0 = (uint32_t)(pF.x & 0x000003FF);
        y1 = (uint32_t)(pF.w & 0x000003FF);

        pF.y = (uint32_t)p_Src[j + k + 10] + ((uint32_t)p_Src[j + k + 11] << 16);
        pF.z = (uint32_t)p_Src[j + k + 12];

        y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20);
        u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
        v1 = (uint32_t)(pF.z & 0x000003FF);

        pF.x = (uint32_t)p_Src[j + k + 16] + ((uint32_t)p_Src[j + k + 17] << 16);
        pF.z = ((uint32_t)p_Src[j + k + 21] << 16);
        pF.w = (uint32_t)p_Src[j + k + 22] + ((uint32_t)p_Src[j + k + 23] << 16);

        y3 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
        u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
        v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
        y4 = (uint32_t)(pF.w & 0x000003FF);

        pF.y = ((uint32_t)p_Src[j + k + 27] << 16);

        y5 = (uint32_t)((pF.y & 0x3FF00000) >> 20);

        k = tid * 6;
        j = tidd * nDstWidth;
        p_dst[j + k + 0] = y0;
        p_dst[j + k + 1] = y1;
        p_dst[j + k + 2] = y2;
        p_dst[j + k + 3] = y3;
        p_dst[j + k + 4] = y4;
        p_dst[j + k + 5] = y5;
        k = tid * 3;
        j = tidd * nDstWidth / 2 + nDstWidth * nDstHeight;
        p_dst[j + k + 0] = u0;
        p_dst[j + k + 1] = u1;
        p_dst[j + k + 2] = u2;
        j = tidd * nDstWidth / 2 + nDstWidth * nDstHeight * 3 / 2;
        p_dst[j + k + 0] = v0;
        p_dst[j + k + 1] = v1;
        p_dst[j + k + 2] = v2;
    }
}

void resizeBatch(uint16_t *dpSrc, int nSrcPitch, int nSrcHeight, uint16_t *dpDst, int nDstWidth, int nDstHeight,
    int nBatch, cudaStream_t stram) {
    dim3 blocks(32, 16, 1);
    dim3 grids((nSrcPitch + blocks.x - 1) / blocks.x, (((nSrcHeight * 4 / 3) + blocks.y) - 1) / blocks.y, 1);
    resizeBatchKernel << <grids, blocks, 0, stram >> > (dpSrc, nSrcPitch, nSrcHeight, dpDst, nDstWidth, nDstHeight, nBatch);
}
