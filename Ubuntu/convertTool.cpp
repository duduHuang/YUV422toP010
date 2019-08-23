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

int parseCmdLine(nv210_context_t *g_ctx) {
    // Run using default arguments
    g_ctx->input_v210_file = "v210.yuv";
    if (g_ctx->input_v210_file == NULL) {
        cout << "Cannot find input file\n Exiting\n";
        return -1;
    }
    return EXIT_SUCCESS;
}

static int loadV210Frame(unsigned short *dSrc, nv210_context_t *g_ctx) {
    int frameSize = g_ctx->img_rowByte * g_ctx->img_height;
    ifstream v210File(g_ctx->input_v210_file, ifstream::in | ios::binary);
    if (!v210File.is_open()) {
        cerr << "Can't open files\n";
        return -1;
    }
    v210File.read((char *)dSrc, frameSize * sizeof(unsigned short));
    if (v210File.gcount() < frameSize * sizeof(unsigned short)) {
        cerr << "can't get one frame\n";
        return -1;
    }
    v210File.close();
    return EXIT_SUCCESS;
}

void ConverterTool::lookupTableF() {
    lookupTable = new int[1024];

    cudaStatus = cudaMalloc((void**)&lookupTable_cuda, sizeof(int) * 1024);
    if (cudaStatus != cudaSuccess) {
        cerr << "lookupTable_cuda cudaMalloc failed!\n";
        freeMemory();
        return;
    }

    for (int i = 0; i < 1024; ++i) {
        lookupTable[i] = round(i * 0.249);
    }
    checkCudaErrors(cudaMemcpy(lookupTable_cuda, lookupTable, sizeof(int) * 1024, cudaMemcpyHostToDevice));
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
    g_ctx->device = findCudaDevice(argc, (const char **)argv);

    // check the compute capability of the device
    int num_devices = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));

    if (0 == num_devices) {
        printf("your system does not have a CUDA capable device, waiving test...\n");
        return false;
    }

    // check if the command-line chosen device ID is within range, exit if not
    if (g_ctx->device >= num_devices) {
        printf("cuda_device=%d is invalid, must choose device ID between 0 and %d\n", g_ctx->device, num_devices - 1);
        return false;
    }

    checkCudaErrors(cudaSetDevice(g_ctx->device));

    // Checking for compute capabilities
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, g_ctx->device));

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

inline void AllocateHostMemory(bool bPinGenericMemory, unsigned short **pp_a, unsigned short **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    if (bPinGenericMemory) {
        // allocate a generic page-aligned chunk of system memory
#ifdef WIN32
        cout << "> VirtualAlloc() allocating " << (float)nbytes / 1048576.0f
            << " Mbytes of (generic page-aligned system memory)\n";
        *pp_a = (unsigned short *)VirtualAlloc(NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        cout << "> mmap() allocating " << (float)nbytes / 1048576.0f << " Mbytes (generic page-aligned system memory)\n";
        *pp_a = (unsigned short *)mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
        *ppAligned_a = (unsigned short *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);
        cout << "> cudaHostRegister() registering " << (float)nbytes / 1048576.0f
            << " Mbytes of generic allocated system memory\n";
        // pin allocate memory
        checkCudaErrors(cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
    }
    else {
#endif
#endif
        cout << "> cudaMallocHost() allocating " << (float)nbytes / 1048576.0f << " Mbytes of system memory\n";
        // allocate host memory (pinned is required for achieve asynchronicity)
        checkCudaErrors(cudaMallocHost((void **)pp_a, nbytes));
        *ppAligned_a = *pp_a;
    }
}

inline void FreeHostMemory(bool bPinGenericMemory, unsigned short **pp_a, unsigned short **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    // CUDA 4.0 support pinning of generic host memory
    if (bPinGenericMemory)  {
        // unpin and delete host memory
        checkCudaErrors(cudaHostUnregister(*ppAligned_a));
#ifdef WIN32
        VirtualFree(*pp_a, 0, MEM_RELEASE);
#else
        munmap(*pp_a, nbytes);
#endif
    }
    else {
#endif
#endif
        cudaFreeHost(*pp_a);
    }
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
    checkCudaErrors(cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0)));
    // allocate and initialize an array of stream handles
    int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync : cudaEventDefault);
    checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
    checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));
    //streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    streams = new cudaStream_t[nstreams];
    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // initialize nvjpeg structures
    checkCudaErrors(nvjpegCreateSimple(&en_params->nv_handle));
    checkCudaErrors(nvjpegEncoderStateCreate(en_params->nv_handle, &en_params->nv_enc_state, streams[0]));
    checkCudaErrors(nvjpegEncoderParamsCreate(en_params->nv_handle, &en_params->nv_enc_params, streams[0]));
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
    checkCudaErrors(cudaMalloc((void **)&en_params->t_16, g_ctx->img_rowByte * g_ctx->img_height * sizeof(unsigned short)));
    en_params->nv_image.pitch[0] = g_ctx->dst_width * sizeof(unsigned char);
    en_params->nv_image.pitch[1] = g_ctx->dst_width / 2 * sizeof(unsigned char);
    en_params->nv_image.pitch[2] = g_ctx->dst_width / 2 * sizeof(unsigned char);
    int size = g_ctx->dst_width * g_ctx->dst_height;
    checkCudaErrors(cudaMalloc(&en_params->nv_image.channel[0], size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&en_params->nv_image.channel[1], size / 2 * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&en_params->nv_image.channel[2], size / 2 * sizeof(unsigned char)));
}

void ConverterTool::convertToP208ThenResize(unsigned short *src, unsigned char *p208Dst, int *nJPEGSize) {
    size_t length = 0;

    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(en_params->nv_enc_params, NVJPEG_CSS_422, streams[0]));

    checkCudaErrors(cudaMemcpy((void *)en_params->t_16, (void *)src,
        g_ctx->img_rowByte * g_ctx->img_height * sizeof(unsigned short), cudaMemcpyHostToDevice));
    resizeBatch(en_params->t_16, g_ctx->img_rowByte, g_ctx->img_height,
        en_params->nv_image.channel[0], en_params->nv_image.channel[1], en_params->nv_image.channel[2],
        g_ctx->dst_width, g_ctx->dst_height, lookupTable_cuda, streams[0]);

    checkCudaErrors(nvjpegEncodeYUV(en_params->nv_handle, en_params->nv_enc_state, en_params->nv_enc_params,
        &en_params->nv_image, NVJPEG_CSS_422, g_ctx->dst_width, g_ctx->dst_height, streams[0]));

    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    // get compressed stream size
    checkCudaErrors(
        nvjpegEncodeRetrieveBitstream(en_params->nv_handle, en_params->nv_enc_state, NULL, &length, streams[0]));

    // get stream itself
    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    vector<unsigned char> jpeg(length);
    checkCudaErrors(
        nvjpegEncodeRetrieveBitstream(en_params->nv_handle, en_params->nv_enc_state, jpeg.data(), &length, streams[0]));

    // write stream to file
    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    memcpy(p208Dst, jpeg.data(), length);
    *nJPEGSize = length;
}

void ConverterTool::testFunction() {
    unsigned char *p208;
    p208 = new unsigned char[1280 * 720 * 2];
    int nJPEGSize = 0;
    convertToP208ThenResize(v210Src, p208, &nJPEGSize);
    ofstream output_file("r.jpg", ios::out | ios::binary);
    output_file.write((char *)p208, nJPEGSize);
    output_file.close();
    delete[] p208;
}

int ConverterTool::preprocess() {
    if (parseCmdLine(g_ctx) < 0) {
        return EXIT_FAILURE;
    }

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &v210Src, &v210SrcAligned, v210Size * sizeof(unsigned short));

    // Load v210 yuv data into v210Src
    if (loadV210Frame(v210Src, g_ctx)) {
        cerr << "failed to load data!\n";
        return EXIT_FAILURE;
    }
}

void ConverterTool::freeMemory() {
    cout << "Free memory...\n";
    checkCudaErrors(cudaFree(lookupTable_cuda));
    delete[] lookupTable;

    checkCudaErrors(cudaFree(en_params->t_16));
    checkCudaErrors(cudaFree(en_params->nv_image.channel[0]));
    checkCudaErrors(cudaFree(en_params->nv_image.channel[1]));
    checkCudaErrors(cudaFree(en_params->nv_image.channel[2]));
}

void ConverterTool::destroyCudaEvent() {
    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

    checkCudaErrors(nvjpegEncoderStateDestroy(en_params->nv_enc_state));
    checkCudaErrors(nvjpegEncoderParamsDestroy(en_params->nv_enc_params));
    checkCudaErrors(nvjpegDestroy(en_params->nv_handle));
    delete en_params;
}