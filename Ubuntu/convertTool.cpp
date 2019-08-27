#include "convertTool.h"
#include "convertToP208.h"
#include "convertToRGB.h"
#include "resize.h"

static const char *sSyncMethod[] = {
    "0 (Automatic Blocking)",
    "1 (Spin Blocking)",
    "2 (Yield Blocking)",
    "3 (Undefined Blocking Method)",
    "4 (Blocking Sync Event) = low CPU utilization",
    NULL
};

const char *sDeviceSyncMethod[] = {
    "cudaDeviceScheduleAuto",
    "cudaDeviceScheduleSpin",
    "cudaDeviceScheduleYield",
    "INVALID",
    "cudaDeviceScheduleBlockingSync",
    NULL
};

bool ConverterTool::printError(string func, string api) {
    if (cudaStatus != cudaSuccess) {
        cerr  << func << " " << api << " failed!\n";
        cerr << "CUDA error at " << cudaGetErrorName(cudaStatus) << "\n";
        return false;
    }
    return true;
}

bool ConverterTool::printNVJPEGError(string func, string api) {
    if (nvjpegStatus != NVJPEG_STATUS_SUCCESS) {
        cerr  << func << " " << api << " failed!\n";
        cerr << "CUDA error at " << _cudaGetErrorEnum(nvjpegStatus) << "\n";
        return false;
    }
    return true;
}

void ConverterTool::lookupTableF() {
    lookupTable = new int[1024];

    cudaStatus = cudaMalloc((void**)&lookupTable_cuda, sizeof(int) * 1024);
    if (!printError("lookupTableF", "cudaMalloc")) {
        goto Error;
    }

    for (int i = 0; i < 1024; ++i) {
        lookupTable[i] = round(i * 0.249);
    }
    cudaStatus = cudaMemcpy(lookupTable_cuda, lookupTable, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    if (!printError("lookupTableF", "cudaMemcpy")) {
        goto Error;
    }
Error:
    delete[] lookupTable;
}

bool ConverterTool::isGPUEnable() {
    float scale_factor = 1.0f;
    v210Size = ((7680 + 47) / 48 * 128 / 2) * 4320;

    if ((device_sync_method = getCmdLineArgumentInt(argc, (const char **)argv, "sync_method")) >= 0) {
        if (device_sync_method == 0 || device_sync_method == 1 || device_sync_method == 2 || device_sync_method == 4) {
            cout << "Device synchronization method set to = " << sSyncMethod[device_sync_method] << "\n";
            //printf("Setting reps to 100 to demonstrate steady state\n");
            //nreps = 100;
        }
        else {
            cout << "Invalid command line option sync_method=\"" << device_sync_method << "\\\n";
            return false;
        }
    }
    else {
        return false;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "use_generic_memory")) {
#if defined(__APPLE__) || defined(MACOSX)
        bPinGenericMemory = false;  // Generic Pinning of System Paged memory not currently supported on Mac OSX
#else
        bPinGenericMemory = true;
#endif
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "use_cuda_malloc_host")) {
        bPinGenericMemory = false;
    }

    printf("\n> ");

    // check the compute capability of the device
    int num_devices = 0;
    cudaStatus = cudaGetDeviceCount(&num_devices);
    if (!printError("isGPUEnable", "cudaGetDeviceCount")) {
        return false;
    }

    if (0 == num_devices) {
        printf("your system does not have a CUDA capable device, waiving test...\n");
        return false;
    }

    g_ctx->device = findCudaDevice(argc, (const char **)argv);
    // check if the command-line chosen device ID is within range, exit if not
    if (g_ctx->device >= num_devices) {
        printf("cuda_device=%d is invalid, must choose device ID between 0 and %d\n", g_ctx->device, num_devices - 1);
        return false;
    }

    cudaStatus = cudaSetDevice(g_ctx->device);
    if (!printError("isGPUEnable", "cudaSetDevice")) {
        return false;
    }

    // Checking for compute capabilities
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, g_ctx->device);
    if (!printError("isGPUEnable", "cudaGetDeviceProperties")) {
        return false;
    }

    // Check if GPU can map host memory (Generic Method), if not then we override bPinGenericMemory to be false
    if (bPinGenericMemory) {
        printf("Device: <%s> canMapHostMemory: %s\n", deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");

        if (deviceProp.canMapHostMemory == 0)
        {
            printf("Using cudaMallocHost, CUDA device does not support mapping of generic host memory\n");
            bPinGenericMemory = false;
        }
    }

    // Anything that is less than 32 Cores will have scaled down workload
    scale_factor = max((32.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);
    v210Size = (int)rint((float)v210Size / scale_factor);

    printf("> CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);
    printf("> %d Multiprocessor(s) x %d (Cores/Multiprocessor) = %d (Cores)\n",
        deviceProp.multiProcessorCount,
        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    printf("> scale_factor = %1.4f\n", 1.0f / scale_factor);
    printf("> array_size   = %d\n\n", v210Size);

    // enable use of blocking sync, to reduce CPU usage
    printf("> Using CPU/GPU Device Synchronization method (%s)\n", sDeviceSyncMethod[device_sync_method]);
    return true;
}

ConverterTool::ConverterTool() {
    argc = 0;
    argv = 0;
    v210Size = 0;
    nstreams = 1;
    bPinGenericMemory = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
    device_sync_method = cudaDeviceScheduleAuto; // by default we use BlockingSync
    g_ctx = new nv210_context_t();
    en_params = new encode_params_t();
}

ConverterTool::~ConverterTool() {
    freeMemory();
    delete g_ctx;
    destroyCudaEvent();
}

void ConverterTool::initialCuda() {
    cudaStatus = cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0));
    if (!printError("initialCuda", "cudaSetDeviceFlags")) {
        return;
    }
    // allocate and initialize an array of stream handles
    int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync : cudaEventDefault);
    cudaStatus = cudaEventCreateWithFlags(&start_event, eventflags);
    if (!printError("initialCuda", "cudaEventCreateWithFlags")) {
        return;
    }
    cudaStatus = cudaEventCreateWithFlags(&stop_event, eventflags);
    if (!printError("initialCuda", "cudaEventCreateWithFlags")) {
        return;
    }
    streams = new cudaStream_t[nstreams];
    for (int i = 0; i < nstreams; i++) {
        cudaStatus = cudaStreamCreate(&(streams[i]));
        if (!printError("initialCuda", "cudaStreamCreate")) {
            return;
        }
    }

    // initialize nvjpeg structures
    nvjpegStatus = nvjpegCreateSimple(&en_params->nv_handle);
    if (!printError("initialCuda", "nvjpegCreateSimple")) {
        return;
    }
    nvjpegStatus = nvjpegEncoderStateCreate(en_params->nv_handle, &en_params->nv_enc_state, streams[0]);
    if (!printError("initialCuda", "nvjpegEncoderStateCreate")) {
        return;
    }
    nvjpegStatus = nvjpegEncoderParamsCreate(en_params->nv_handle, &en_params->nv_enc_params, streams[0]);
    if (!printError("initialCuda", "nvjpegEncoderParamsCreate")) {
        return;
    }
}

void ConverterTool::setSrcSize(int w, int h) {
    g_ctx->img_width = w;
    g_ctx->img_height = h;
    g_ctx->img_rowByte = (w + 47) / 48 * 128 / 2;
    v210Size = g_ctx->img_rowByte * g_ctx->img_height;
}

void ConverterTool::setDstSize(int w, int h) {
    g_ctx->dst_width = w;
    g_ctx->dst_height = h;
}

void ConverterTool::allocateMem() {
    cudaStatus = cudaMalloc((void **)&en_params->t_16, g_ctx->img_rowByte * g_ctx->img_height * sizeof(unsigned short));
    if (!printError("allocateMem", "cudaMalloc en_params->t_16")) {
        return;
    }
    en_params->nv_image.pitch[0] = g_ctx->dst_width * sizeof(unsigned char);
    en_params->nv_image.pitch[1] = g_ctx->dst_width / 2 * sizeof(unsigned char);
    en_params->nv_image.pitch[2] = g_ctx->dst_width / 2 * sizeof(unsigned char);
    int size = g_ctx->dst_width * g_ctx->dst_height;
    cudaStatus = cudaMalloc(&en_params->nv_image.channel[0], size * sizeof(unsigned char));
    if (!printError("allocateMem", "en_params->nv_image.channel")) {
        return;
    }
    cudaStatus = cudaMalloc(&en_params->nv_image.channel[1], size / 2 * sizeof(unsigned char));
    if (!printError("allocateMem", "en_params->nv_image.channel")) {
        return;
    }
    cudaStatus = cudaMalloc(&en_params->nv_image.channel[2], size / 2 * sizeof(unsigned char));
    if (!printError("allocateMem", "en_params->nv_image.channel")) {
        return;
    }
}

void ConverterTool::convertToP208ThenResize(unsigned short *src, unsigned char *p208Dst, int *nJPEGSize) {
    size_t length = 0;

    nvjpegStatus = nvjpegEncoderParamsSetSamplingFactors(en_params->nv_enc_params, NVJPEG_CSS_422, streams[0]);
    if (!printError("convertToP208ThenResize", "nvjpegEncoderParamsSetSamplingFactors")) {
        return;
    }

    cudaStatus = cudaMemcpy((void *)en_params->t_16, (void *)src,
        g_ctx->img_rowByte * g_ctx->img_height * sizeof(unsigned short), cudaMemcpyHostToDevice);
    if (!printError("convertToP208ThenResize", "cudaMemcpy en_params->t_16")) {
        return;
    }
    resizeBatch(en_params->t_16, g_ctx->img_rowByte, g_ctx->img_height,
        en_params->nv_image.channel[0], en_params->nv_image.channel[1], en_params->nv_image.channel[2],
        g_ctx->dst_width, g_ctx->dst_height, lookupTable_cuda, streams[0]);

    nvjpegStatus = nvjpegEncodeYUV(en_params->nv_handle, en_params->nv_enc_state, en_params->nv_enc_params,
        &en_params->nv_image, NVJPEG_CSS_422, g_ctx->dst_width, g_ctx->dst_height, streams[0]);
    if (!printError("convertToP208ThenResize", "nvjpegEncodeYUV")) {
        return;
    }

    cudaStatus = cudaStreamSynchronize(streams[0]);
    if (!printError("convertToP208ThenResize", "cudaStreamSynchronize")) {
        return;
    }
    // get compressed stream size
    nvjpegStatus = nvjpegEncodeRetrieveBitstream(en_params->nv_handle, en_params->nv_enc_state, NULL, &length, streams[0]);
    if (!printError("convertToP208ThenResize", "nvjpegEncodeRetrieveBitstream")) {
        return;
    }

    // get stream itself
    cudaStatus = cudaStreamSynchronize(streams[0]);
    if (!printError("convertToP208ThenResize", "cudaStreamSynchronize")) {
        return;
    }
    vector<unsigned char> jpeg(length);
    nvjpegStatus = nvjpegEncodeRetrieveBitstream(en_params->nv_handle, en_params->nv_enc_state, jpeg.data(), &length, streams[0]);
    if (!printError("convertToP208ThenResize", "nvjpegEncodeRetrieveBitstream")) {
        return;
    }

    // write stream to file
    cudaStatus = cudaStreamSynchronize(streams[0]);
    if (!printError("convertToP208ThenResize", "cudaStreamSynchronize")) {
        return;
    }
    memcpy(p208Dst, jpeg.data(), length);
    *nJPEGSize = length;
}

void ConverterTool::allocatSrcMem() {
    int size = g_ctx->img_width * g_ctx->img_height * RGB_SIZE;
    cudaStatus = cudaMalloc((void **)&en_params->t_16, size * sizeof(unsigned short));
    if (!printError("allocatSrcMem", "cudaMalloc en_params->t_16")) {
        return;
    }
}

void ConverterTool::allocatNVJPEGRGBMem() {
    int size = g_ctx->dst_width * g_ctx->dst_height * RGB_SIZE;
    en_params->nv_image.pitch[0] = g_ctx->dst_Width * RGB_SIZE * sizeof(unsigned char);
    cudaStatus = cudaMalloc(&en_params->nv_image.channel[0], size * sizeof(unsigned char));
    if (!printError("allocatNVJPEGRGBMem", "en_params->nv_image.channel[0]")) {
        return;
    }
}

void ConverterTool::allocatV210DstMem() {
    cudaStatus = cudaMalloc((void **)&dev_v210Dst, v210Size * sizeof(unsigned char));
    if (!printError("allocatV210DstMem", "dev_v210Dst")) {
        return;
    }
}

void ConverterTool::RGB10bitConvertToRGB8bitNVJPEG(unsigned short *src, unsigned char *Dst, int *nJPEGSize) {
    size_t length = 0;

    nvjpegStatus = nvjpegEncoderParamsSetSamplingFactors(en_params->nv_enc_params, NVJPEG_CSS_444, streams[0]);
    if (!printError("RGB10bitConvertToRGB8bitNVJPEG", "nvjpegEncoderParamsSetSamplingFactors")) {
        return;
    }

    cudaStatus = cudaMemcpy((void *)en_params->t_16, (void *)src,
        g_ctx->img_width * g_ctx->img_height * 3 * sizeof(unsigned short), cudaMemcpyHostToDevice);
    if (!printError("convertToP208ThenResize", "cudaMemcpy en_params->t_16")) {
        return;
    }
}

void ConverterTool::freeMemory() {
    cout << "Free memory...\n";
    cudaStatus = cudaFree(lookupTable_cuda);
    printError("freeMemory", "cudaFree lookupTable_cuda");

    cudaStatus = cudaFree(en_params->t_16);
    printError("freeMemory", "cudaFree en_params->t_16");

    cudaStatus = cudaFree(en_params->nv_image.channel[0]);
    printError("freeMemory", "cudaFree en_params->nv_image.channel");

    cudaStatus = cudaFree(en_params->nv_image.channel[1]);
    printError("freeMemory", "cudaFree en_params->nv_image.channel");

    cudaStatus = cudaFree(en_params->nv_image.channel[2]);
    printError("freeMemory", "cudaFree en_params->nv_image.channel");
}

void ConverterTool::destroyCudaEvent() {
    for (int i = 0; i < nstreams; i++) {
        cudaStatus = cudaStreamDestroy(streams[i]);
        printError("destroyCudaEvent", "cudaStreamDestroy streams");
    }
    delete[] streams;
    cudaStatus = cudaEventDestroy(start_event);
    printError("destroyCudaEvent", "cudaStreamDestroy streams");

    cudaStatus = cudaEventDestroy(stop_event);
    printError("destroyCudaEvent", "cudaStreamDestroy stop_event");

    nvjpegStatus = nvjpegEncoderStateDestroy(en_params->nv_enc_state);
    printError("destroyCudaEvent", "cudaStreamDestroy nv_enc_state");

    nvjpegStatus = nvjpegEncoderParamsDestroy(en_params->nv_enc_params);
    printError("destroyCudaEvent", "cudaStreamDestroy nv_enc_params");

    nvjpegStatus = nvjpegDestroy(en_params->nv_handle);
    printError("destroyCudaEvent", "cudaStreamDestroy nv_handle");

    delete en_params;
}