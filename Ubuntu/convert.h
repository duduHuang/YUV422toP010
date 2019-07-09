#ifndef __H_CONVERT__
#define __H_CONVERT__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;

#define TEST_LOOP 1

typedef struct _nv210_to_p010_context_t {
	int width;
	int height;
	int device;  // cuda device ID
	int ctx_pitch; // the value will be suitable for Texture memroy.
	int batch;
	char *input_v210_file;
} nv210_to_p010_context_t;

int convert(int argc, char* argv[]);

#endif // !__H_CONVERT__