#ifndef __H_CONVERTTOP010__
#define __H_CONVERTTOP010__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

// v210 to p010
extern "C"
void convertToP010(uint16_t *dpSrc, int nSrcPitch, uint16_t *dpDst, int nDstWidth, int nDstHeight,
	int nBatch, int block_size, cudaStream_t stram = 0);

#endif // !__H_CONVERTTOP010__