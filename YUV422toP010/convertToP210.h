#ifndef __H_CONVERTTOP210__
#define __H_CONVERTTOP210__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

// v210 to p010
extern "C"
void convertToP010(uint16_t *dpSrc, uint16_t *dpDst, int nDstWidth, int nDstHeight,
	int nBatch, cudaStream_t stram = 0);

void convertToP210(uint16_t *dpSrc, uint16_t *dpDst, int nWidth, int nHeight, cudaStream_t stream = 0);

#endif // !__H_CONVERTTOP210__