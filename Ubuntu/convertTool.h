#ifndef __H_CONVERTTOOL__
#define __H_CONVERTTOOL__
#include "nvjpeg.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

#define TEST_LOOP 1
#define RGB_SIZE 3
#define YUV422_PLANAR_SIZE 2
#define DEFAULT_PINNED_GENERIC_MEMORY true

#ifndef WIN32
#include <sys/mman.h> // for mmap() / munmap()
#endif

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x, size) ( ((size_t)x + (size - 1)) & (~(size - 1)) )

typedef struct _nv210_context_t {
    int img_width;
    int img_height;
    int device;  // cuda device ID
    int img_rowByte; // the value will be suitable for Texture memroy.
    int batch;

    int dst_width;
    int dst_height;
    char *input_v210_file;
} nv210_context_t;

typedef struct _encode_params_t {
    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    nvjpegImage_t nv_image;
    nvjpegStatus_t err;

    unsigned short *t_16;
    unsigned char *t_8;
} encode_params_t;

class ConverterTool {
private:
    int argc, v210Size, nstreams;
    char **argv;
    unsigned short *v210Dst, *v210DstAligned, *dev_v210Dst;
    unsigned short *rgb10bitSrc, *rgb10bitSrcAligned, *dev_rgb10bitSrc;
    nv210_context_t *g_ctx;
    encode_params_t *en_params;
    // allocate generic memory and pin it laster instead of using cudaHostAlloc()
    bool bPinGenericMemory; // we want this to be the default behavior
    int device_sync_method; // by default we use BlockingSync
    cudaError_t cudaStatus;
    nvjpegStatus_t nvjpegStatus;
    cudaEvent_t start_event, stop_event;
    cudaStream_t *streams;
    int *lookupTable, *lookupTable_cuda;
public:
    ConverterTool();
    bool printError(string func, string api);
    bool printNVJPEGError(string func, string api);
    bool isGPUEnable();
    void initialCuda();
    void lookupTableF();

    void setSrcSize(int w, int h);
    void setDstSize(int w, int h);

    // IDUDUConvert API
    void allocateMem();
    void convertToP208ThenResize(unsigned short *src, unsigned char *p208Dst, int *nJPEGSize);

    // IDUDURGBConvert API
    void allocatSrcMem();
    void allocatNVJPEGRGBMem();
    void allocatV210DstMem();
    void RGB10bitConvertToRGB8bitNVJPEG(unsigned short *src, unsigned char *Dst, int *nJPEGSize);

    void freeMemory();
    void destroyCudaEvent();
    ~ConverterTool();
};

#endif // !__H_CONVERTTOOL__