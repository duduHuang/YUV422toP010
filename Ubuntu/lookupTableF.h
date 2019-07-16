#ifndef __H_LOOKUPTABLE__
#define __H_LOOKUPTABLE__

#include <iostream>
#include <helper_cuda.h>

using namespace std;

// 10 bit to 8 bit look up table
extern "C"
void lookupTableF(int *lookupTable, cudaStream_t stream = 0);

#endif // !__H_LOOKUPTABLE__