#ifndef __H_CONVERTTORGB__
#define __H_CONVERTTORGB__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

// v210 to rgb
extern "C"
void convertToRGB(uint16_t *dpSrc, int nSrcPitch, uint16_t *dpDst, int nDstWidth, int nDstHeight,
	int nBatch, int block_size, cudaStream_t stram = 0);

#endif // !__H_CONVERTTORGB__