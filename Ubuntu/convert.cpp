#include "convert.h"
#include "convertToP210.h"
#include "convertToRGB.h"
#include "libdpx/DPX.h"
#include "resize.h"

using namespace dpx;

nv210_to_p010_context_t g_ctx;
void converter(unsigned short *v210Src, unsigned char *p208Dst, int nSrcW, int nSrcH, int nDstw, int nDstH, int *nJPEGSize);

//           call this method to do -----> check cuda / gpu
bool isGPUEnable() {
    cout << "\n CUDA Device Query (Runtime API) version (CUDART static linking)\n\n";
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        cout << "Result = FAIL\n\n";
        return false;
    }
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        cout << "There are no available device(s) that support CUDA\n\n";
        return false;
    } else {
        cout << "Detected " << deviceCount << " CUDA Capable device(s)\n\n";
        return true;
    }
}

int parseCmdLine(int argc, char *argv[]) {
    int w = 1280, h = 720;
    string s;
    memset(&g_ctx, 0, sizeof(g_ctx));
    if (argc == 3) {
        // Run using default arguments
        g_ctx.input_v210_file = sdkFindFilePath(argv[1], argv[0]);
        if (g_ctx.input_v210_file == NULL) {
            cout << "Cannot find input file" << argv[1] << "\n Exiting\n";
            return -1;
        }
        g_ctx.width = 7680;
        g_ctx.height = 4320;
        g_ctx.batch = 1;
        cout << "Output resolution: (7680 4320)\n"
             << "                   (3840 2160)\n"
             << "                   (1920 1080)\n"
             << "                   default (1280 720) ";
        getline(cin, s);
        if (!s.empty()) {
            istringstream ss(s);
            ss >> w >> h;
        }
        g_ctx.dst_width = w;
        g_ctx.dst_height = h;
    }
    cout << "\n";
    g_ctx.device = findCudaDevice(argc, (const char **)argv);
    if (g_ctx.width == 0 || g_ctx.height == 0 || !g_ctx.input_v210_file) {
        cout << "Usage: " << argv[0] << " inputf outputf\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/*
  Load v210 yuvfile data into GPU device memory
*/
static int loadV210Frame(unsigned short *d_inputV210) {
    unsigned short *pV210FrameData, *d_V210;
    int frameSize = 0;
    ifstream v210File(g_ctx.input_v210_file, ifstream::in | ios::binary);
    if (!v210File.is_open()) {
        cerr << "Can't open files\n";
        return -1;
    }

    frameSize = g_ctx.ctx_pitch * g_ctx.height;

#if USE_UVM_MEM
    pV210FrameData = d_inputV210;
#else
    pV210FrameData = (unsigned short *)malloc(frameSize * sizeof(unsigned short));
    if (pV210FrameData == NULL) {
        cerr << "Failed to malloc FrameData\n";
        return -1;
    }
    memset((void *)pV210FrameData, 0, frameSize * sizeof(unsigned short));
#endif
    d_V210 = pV210FrameData;
    v210File.read((char *)pV210FrameData, frameSize * sizeof(unsigned short));
    if (v210File.gcount() < frameSize * sizeof(unsigned short)) {
        cerr << "can't get one frame\n";
        return -1;
    }
#if USE_UVM_MEM
    // Prefetch to GPU for following GPU operation
    cudaStreamAttachMemAsync(NULL, pV210FrameData, 0, cudaMemAttachGlobal);
#endif
    d_V210 = d_inputV210;
    //for (int i = 0; i < g_ctx.batch; i++) {
        checkCudaErrors(cudaMemcpy((void *)d_V210, (void *)pV210FrameData,
            frameSize * sizeof(unsigned short), cudaMemcpyHostToDevice));
    //  d_V210 += frameSize;
    //}
#if (USE_UVM_MEM == 0)
    free(pV210FrameData);
#endif
    v210File.close();
    return EXIT_SUCCESS;
}

void lookupTableF(int *lookupTable) {
    for (int i = 0; i < 1024; ++i) {
        lookupTable[i] = round(i * 0.249);
    }
}

/*
  DPX header
*/
void dpxHandler(unsigned short *src, int width, int height, int batch, string s, int coloer_space) {
    unsigned short *nv12Data, *d_nv12;
    int size = 0;
    OutStream img;
    Writer writerf;
    char *fileName = new char[s.length() + 1];
    strcpy(fileName, s.c_str());

    size = width * height * coloer_space;

    nv12Data = (unsigned short *)malloc(size * sizeof(unsigned short));
    if (nv12Data == NULL) {
        cerr << "Failed to allcoate memory\n";
        return;
    }
    memset((void *)nv12Data, 0, size * sizeof(unsigned short));
    d_nv12 = src;
    cudaMemcpy((void *)nv12Data, (void *)d_nv12, size * sizeof(unsigned short), cudaMemcpyDeviceToHost);

    if (!img.Open(fileName)) {
        cout << "Unable to open file " << fileName << endl;
        return;
    }
    writerf.header.Reset();
    writerf.SetOutStream(&img);
    writerf.header.SetVersion(SMPTE_VERSION);
    writerf.header.SetNumberOfElements(1);
    writerf.header.SetImageEncoding(0, kNone);
    writerf.header.SetFileSize(size * sizeof(unsigned short));
    writerf.header.SetBitDepth(0, 16);
    writerf.header.SetBlackGain(0);
    writerf.header.SetBlackLevel(0);
    writerf.header.SetColorimetric(0, kUserDefined);
    writerf.header.SetVerticalSampleRate(0);
    writerf.header.SetFieldNumber(0);
    writerf.header.SetInterlace(0);
    writerf.header.SetHeldCount(0);
    writerf.header.SetSequenceLength(0);
    writerf.header.SetFramePosition(0);
    writerf.header.SetGamma(0);
    writerf.header.SetImageOffset(8192);
    writerf.header.SetDataSign(0, 0);
    writerf.header.SetLowData(0, 0);
    writerf.header.SetLowQuantity(0, 0);
    writerf.header.SetHighData(0, 0);
    writerf.header.SetHighQuantity(0, 0);
    writerf.header.SetImageDescriptor(0, kRGB);
    writerf.header.SetDataOffset(0, 8192);
    writerf.header.SetEndOfImagePadding(0, 0);
    writerf.header.SetEndOfLinePadding(0, 0);
    writerf.header.SetDittoKey(1);
    writerf.header.SetBorder(0, 0);
    writerf.header.SetBorder(1, 0);
    writerf.header.SetBorder(2, 0);
    writerf.header.SetBorder(3, 0);
    writerf.header.SetHorizontalSampleRate(0);
    writerf.header.SetShutterAngle(0);
    writerf.header.SetFrameRate(0);
    writerf.header.SetXCenter(0);
    writerf.header.SetXOffset(0);
    writerf.header.SetXOriginalSize(0);
    writerf.header.SetXScannedSize(0);
    writerf.header.SetYCenter(0);
    writerf.header.SetYOffset(0);
    writerf.header.SetYOriginalSize(0);
    writerf.header.SetYScannedSize(0);
    writerf.header.SetEncryptKey(0xFFFFFFFF);
    writerf.header.SetTransfer(0, kUserDefined);
    writerf.header.SetImagePacking(0, kPacked);
    writerf.header.SetUserSize(0);
    writerf.header.SetAspectRatio(0, 0);
    writerf.header.SetAspectRatio(1, 0);
    writerf.header.SetBreakPoint(0);
    writerf.header.SetWhiteLevel(0);
    writerf.header.SetIntegrationTimes(0);
    writerf.header.SetTimeOffset(0);
    writerf.header.SetTemporalFrameRate(0);
    writerf.header.SetTimeCode("00:00:00:00");
    writerf.header.SetUserBits("00:00:00:00");
    writerf.header.SetCreationTimeDate("2019:07:05:15:30:00");
    writerf.header.SetImageOrientation(kLeftToRightTopToBottom);
    writerf.header.SetLinesPerElement(g_ctx.height);
    writerf.header.SetPixelsPerLine(g_ctx.width);
    writerf.WriteHeader();
    writerf.WriteElement(0, (void *)nv12Data);
    writerf.Finish();

    img.Close();
    free(nv12Data);
}

void dpxHandler(unsigned char *src, int width, int height, int batch, string s, int coloer_space) {
    unsigned char *nv12Data, *d_nv12;
    int size = 0;
    OutStream img;
    Writer writerf;
    char *fileName = new char[s.length() + 1];
    strcpy(fileName, s.c_str());

    size = width * height * coloer_space;

    nv12Data = (unsigned char *)malloc(size * sizeof(unsigned char));
    if (nv12Data == NULL) {
        cerr << "Failed to allcoate memory\n";
        return;
    }
    memset((void *)nv12Data, 0, size * sizeof(unsigned char));
    d_nv12 = src;
    cudaMemcpy((void *)nv12Data, (void *)d_nv12, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    if (!img.Open(fileName)) {
        cout << "Unable to open file " << fileName << endl;
        return;
    }
    writerf.header.Reset();
    writerf.SetOutStream(&img);
    writerf.header.SetVersion(SMPTE_VERSION);
    writerf.header.SetNumberOfElements(1);
    writerf.header.SetImageEncoding(0, kNone);
    writerf.header.SetFileSize(size * sizeof(unsigned char));
    writerf.header.SetBitDepth(0, 16);
    writerf.header.SetBlackGain(0);
    writerf.header.SetBlackLevel(0);
    writerf.header.SetColorimetric(0, kUserDefined);
    writerf.header.SetVerticalSampleRate(0);
    writerf.header.SetFieldNumber(0);
    writerf.header.SetInterlace(0);
    writerf.header.SetHeldCount(0);
    writerf.header.SetSequenceLength(0);
    writerf.header.SetFramePosition(0);
    writerf.header.SetGamma(0);
    writerf.header.SetImageOffset(8192);
    writerf.header.SetDataSign(0, 0);
    writerf.header.SetLowData(0, 0);
    writerf.header.SetLowQuantity(0, 0);
    writerf.header.SetHighData(0, 0);
    writerf.header.SetHighQuantity(0, 0);
    writerf.header.SetImageDescriptor(0, kRGB);
    writerf.header.SetDataOffset(0, 8192);
    writerf.header.SetEndOfImagePadding(0, 0);
    writerf.header.SetEndOfLinePadding(0, 0);
    writerf.header.SetDittoKey(1);
    writerf.header.SetBorder(0, 0);
    writerf.header.SetBorder(1, 0);
    writerf.header.SetBorder(2, 0);
    writerf.header.SetBorder(3, 0);
    writerf.header.SetHorizontalSampleRate(0);
    writerf.header.SetShutterAngle(0);
    writerf.header.SetFrameRate(0);
    writerf.header.SetXCenter(0);
    writerf.header.SetXOffset(0);
    writerf.header.SetXOriginalSize(0);
    writerf.header.SetXScannedSize(0);
    writerf.header.SetYCenter(0);
    writerf.header.SetYOffset(0);
    writerf.header.SetYOriginalSize(0);
    writerf.header.SetYScannedSize(0);
    writerf.header.SetEncryptKey(0xFFFFFFFF);
    writerf.header.SetTransfer(0, kUserDefined);
    writerf.header.SetImagePacking(0, kPacked);
    writerf.header.SetUserSize(0);
    writerf.header.SetAspectRatio(0, 0);
    writerf.header.SetAspectRatio(1, 0);
    writerf.header.SetBreakPoint(0);
    writerf.header.SetWhiteLevel(0);
    writerf.header.SetIntegrationTimes(0);
    writerf.header.SetTimeOffset(0);
    writerf.header.SetTemporalFrameRate(0);
    writerf.header.SetTimeCode("00:00:00:00");
    writerf.header.SetUserBits("00:00:00:00");
    writerf.header.SetCreationTimeDate("2019:07:05:15:30:00");
    writerf.header.SetImageOrientation(kLeftToRightTopToBottom);
    writerf.header.SetLinesPerElement(g_ctx.height);
    writerf.header.SetPixelsPerLine(g_ctx.width);
    writerf.WriteHeader();
    writerf.WriteElement(0, (void *)nv12Data);
    writerf.Finish();

    img.Close();
    free(nv12Data);
}

/*
  Draw yuv data
*/
void dumpYUV(unsigned short *d_srcNv12, int width, int height, int batch, string filename, int coloer_space, cudaStream_t stream) {
    unsigned short *nv12Data, *d_nv12;
    int size = 0;
    size = width * height * coloer_space;

    nv12Data = (unsigned short *)malloc(size * sizeof(unsigned short));
    if (nv12Data == NULL) {
        cerr << "Failed to allcoate memory\n";
        return;
    }
    memset((void *)nv12Data, 0, size * sizeof(unsigned short));
    d_nv12 = d_srcNv12;
    for (int i = 0; i < batch; i++) {
        ofstream nv12File(filename, ostream::out | ostream::binary);

        cudaMemcpy((void *)nv12Data, (void *)d_nv12, size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        if (nv12File) {
            nv12File.write((char *)nv12Data, size * sizeof(unsigned short));
            nv12File.close();
        }
        d_nv12 += width * height;
    }
    free(nv12Data);
}

void dumpYUV(unsigned char *d_srcNv12, int width, int height, int batch, string filename, int coloer_space, cudaStream_t stream) {
    unsigned char *nv12Data, *d_nv12;
    int size = 0;
    size = width * height * coloer_space;

    nv12Data = (unsigned char *)malloc(size * sizeof(unsigned char));
    if (nv12Data == NULL) {
        cerr << "Failed to allcoate memory\n";
        return;
    }
    memset((void *)nv12Data, 0, size * sizeof(unsigned char));
    d_nv12 = d_srcNv12;
    for (int i = 0; i < batch; i++) {
        ofstream nv12File(filename, ostream::out | ostream::binary);

        cudaMemcpy((void *)nv12Data, (void *)d_nv12, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        if (nv12File) {
            nv12File.write((char *)nv12Data, size * sizeof(unsigned char));
            nv12File.close();
        }
        d_nv12 += width * height;
    }
    free(nv12Data);
}

/*
  Jpeg encoder
*/
int encode_images(unsigned short *d_inputV210, char *filename, int width, int height, int batch, int dstWidth, int dstHeight,
    int *lookupTable_cuda, int n, cudaStream_t stream) {
    encode_params_t en_params;
    size_t length = 0;
    char c = 'n';
    string s_cin;
    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsedTime = 0.0f;
    int rowByte = (width + 47) / 48 * 128 / 2;

    // initialize nvjpeg structures
    checkCudaErrors(nvjpegCreateSimple(&en_params.nv_handle));
    checkCudaErrors(nvjpegEncoderStateCreate(en_params.nv_handle, &en_params.nv_enc_state, stream));
    checkCudaErrors(nvjpegEncoderParamsCreate(en_params.nv_handle, &en_params.nv_enc_params, stream));

    // Set encode parameters
    //checkCudaErrors(nvjpegEncoderParamsSetQuality(en_params.nv_enc_params, 70, stream));
    //checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(en_params.nv_enc_params, 0, stream));

    if (n == 1) {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(en_params.nv_enc_params, NVJPEG_CSS_444, stream));
        en_params.nv_image.pitch[0] = dstWidth * RGB_SIZE * sizeof(unsigned char);
        checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[0], dstWidth * dstHeight * RGB_SIZE * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc(&en_params.t_8, width * height * RGB_SIZE * sizeof(unsigned char)));
        cudaEventRecord(start, 0);
        convertToRGB(d_inputV210, en_params.t_8, rowByte, width, height, batch, lookupTable_cuda, PACKED, stream);
        resizeBatch(en_params.t_8, width, height, en_params.nv_image.channel[0], dstWidth, dstHeight, batch, stream);
        checkCudaErrors(nvjpegEncodeImage(en_params.nv_handle, en_params.nv_enc_state, en_params.nv_enc_params,
            &en_params.nv_image, NVJPEG_INPUT_RGBI, dstWidth, dstHeight, stream));
    } else if (n == 2) {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(en_params.nv_enc_params, NVJPEG_CSS_444, stream));
        en_params.nv_image.pitch[0] = dstWidth * RGB_SIZE * sizeof(unsigned char);
        checkCudaErrors(cudaMalloc((void **)&en_params.t_16, dstWidth * dstHeight * YUV422_PLANAR_SIZE * sizeof(unsigned short)));
        checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[0], dstWidth * dstHeight * RGB_SIZE * sizeof(unsigned char)));
        cudaEventRecord(start, 0);
        resizeBatch(d_inputV210, rowByte, height, en_params.t_16, dstWidth, dstHeight, batch, stream);
        convertToRGB(en_params.t_16, en_params.nv_image.channel[0], dstWidth, dstWidth, dstHeight, batch, lookupTable_cuda, PLANAR, stream);
        checkCudaErrors(nvjpegEncodeImage(en_params.nv_handle, en_params.nv_enc_state, en_params.nv_enc_params,
            &en_params.nv_image, NVJPEG_INPUT_RGBI, dstWidth, dstHeight, stream));
    } else if (n == 3) {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(en_params.nv_enc_params, NVJPEG_CSS_422, stream));
        en_params.nv_image.pitch[0] = dstWidth * sizeof(unsigned char);
        en_params.nv_image.pitch[1] = dstWidth / 2 * sizeof(unsigned char);
        en_params.nv_image.pitch[2] = dstWidth / 2 * sizeof(unsigned char);
        checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[0], dstWidth * dstHeight * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[1], dstWidth * dstHeight / 2 * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[2], dstWidth * dstHeight / 2 * sizeof(unsigned char)));
        cudaEventRecord(start, 0);
        resizeBatch(d_inputV210, rowByte, height,
            en_params.nv_image.channel[0], en_params.nv_image.channel[1], en_params.nv_image.channel[2],
            dstWidth, dstHeight, batch, lookupTable_cuda, stream);
        checkCudaErrors(nvjpegEncodeYUV(en_params.nv_handle, en_params.nv_enc_state, en_params.nv_enc_params,
            &en_params.nv_image, NVJPEG_CSS_422, dstWidth, dstHeight, stream));
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
    // get compressed stream size
    checkCudaErrors(
      nvjpegEncodeRetrieveBitstream(en_params.nv_handle, en_params.nv_enc_state, NULL, &length, stream));

    // get stream itself
    checkCudaErrors(cudaStreamSynchronize(stream));
    vector<unsigned char> jpeg(length);
    checkCudaErrors(nvjpegEncodeRetrieveBitstream(en_params.nv_handle, en_params.nv_enc_state, jpeg.data(), &length, stream));

    // write stream to file
    checkCudaErrors(cudaStreamSynchronize(stream));
    ofstream output_file(filename, ios::out | ios::binary);
    output_file.write((char *)jpeg.data(), length);
    output_file.close();
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout << fixed << "  CUDA v210 to nvJPEG(" << width << "x" << height << " --> "
        << dstWidth << "x" << dstHeight << "), "
        << "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
        << " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / batch << " ms/frame\n";
    c = 'n';
    s_cin.clear();
    cout << "Write the dpx file ? (y/N) ";
    getline(cin, s_cin);
    if (!s_cin.empty()) {
        istringstream ss(s_cin);
        ss >> c;
    }
    if (c == 'y') {
        string s;
        cout << "File name : ";
        cin >> s;
        if (n == 1 || n == 2) {
            dpxHandler(en_params.nv_image.channel[0], dstWidth, dstHeight, batch, s, RGB_SIZE);
        } else if (n == 3) {

        }
    }
    c = 'n';
    s_cin.clear();
    cout << "Write the row file ? (y/N) ";
    getline(cin, s_cin);
    if (!s_cin.empty()) {
        istringstream ss(s_cin);
        ss >> c;
    }
    if (c == 'y') {
        string s;
        cout << "File name : ";
        cin >> s;
        if (n == 1 || n == 2) {
            dumpYUV(en_params.nv_image.channel[0], dstWidth, dstHeight, batch, s, RGB_SIZE, stream);
        } else if (n == 3) {

        }
    }
    if (n == 1) {
        checkCudaErrors(cudaFree(en_params.nv_image.channel[0]));
        checkCudaErrors(cudaFree(en_params.t_8));
    } else if (n == 2) {
        checkCudaErrors(cudaFree(en_params.nv_image.channel[0]));
        checkCudaErrors(cudaFree(en_params.t_16));
    } else if (n == 3) {
        checkCudaErrors(cudaFree(en_params.nv_image.channel[0]));
        checkCudaErrors(cudaFree(en_params.nv_image.channel[1]));
        checkCudaErrors(cudaFree(en_params.nv_image.channel[2]));
    }
    checkCudaErrors(nvjpegDestroy(en_params.nv_handle));
    checkCudaErrors(nvjpegEncoderStateDestroy(en_params.nv_enc_state));
    checkCudaErrors(nvjpegEncoderParamsDestroy(en_params.nv_enc_params));
    return EXIT_SUCCESS;
}

/*
  Convert v210 to p010
*/
void v210ToP010(unsigned short *d_inputV210, char *argv) {
    unsigned short *d_outputYUV422, *d_outputRGB10;
    unsigned char *d_outputRGB8, *d_outputYUV422_1;
    int size = 0, n = 3;
    int *lookupTable, *lookupTable_cuda;
    char c = 'y';
    string s_cin;
    size = g_ctx.width * g_ctx.height * g_ctx.batch;
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsedTime = 0.0f;

    c ='y';
    s_cin.clear();
    cout << "Initial look up table ? (Y/n) ";
    getline(cin, s_cin);
    if (!s_cin.empty()) {
        istringstream ss(s_cin);
        ss >> c;
    }
    if (c == 'y') {
        lookupTable = (int *)malloc(sizeof(int) * 1024);
        checkCudaErrors(cudaMalloc((void**)&lookupTable_cuda, sizeof(int) * 1024));
        lookupTableF(lookupTable);
        cudaMemcpy(lookupTable_cuda, lookupTable, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    } else {
        lookupTable = (int *)malloc(sizeof(int) * 1024);
        checkCudaErrors(cudaMalloc((void**)&lookupTable_cuda, sizeof(int) * 1024));
        memset((void *)lookupTable, 0, 1024 * sizeof(int));
        cudaMemcpy(lookupTable_cuda, lookupTable, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    }
    n = 3;
    s_cin.clear();
    cout << "Convert to 1. P210\n"
         << "           2. RGB\n"
         << "           3. nvJPEG ? (1, 2, (default) 3) ";
    getline(cin, s_cin);
    if (!s_cin.empty()) {
        istringstream ss(s_cin);
        ss >> n;
    }
    if (n == 1) {
        cout << "10 bit to 8 bit ? (y/N) ";
        c = 'n';
        s_cin.clear();
        getline(cin, s_cin);
        if (!s_cin.empty()) {
            istringstream ss(s_cin);
            ss >> c;
        }
        if (c == 'y') {
            checkCudaErrors(cudaMalloc((void **)&d_outputYUV422_1, size * YUV422_PLANAR_SIZE * sizeof(unsigned char)));
            cudaEventRecord(start, 0);
            for (int i = 0; i < TEST_LOOP; i++) {
                convertToP208(d_inputV210, d_outputYUV422_1, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height, lookupTable_cuda, stream);
            }
        } else if (c == 'n') {
            checkCudaErrors(cudaMalloc((void **)&d_outputYUV422, size * YUV422_PLANAR_SIZE * sizeof(unsigned short)));
            cudaEventRecord(start, 0);
            for (int i = 0; i < TEST_LOOP; i++) {
                convertToP210(d_inputV210, d_outputYUV422, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height, stream);
            }
        }
    } else if (n == 2) {
        cout << "10 bit to 8 bit ? (y/N) ";
        c = 'n';
        s_cin.clear();
        getline(cin, s_cin);
        if (!s_cin.empty()) {
            istringstream ss(s_cin);
            ss >> c;
        }
        if (c == 'y') {
            checkCudaErrors(cudaMalloc((void **)&d_outputRGB8, size * RGB_SIZE * sizeof(unsigned char)));
            cudaEventRecord(start, 0);
            for (int i = 0; i < TEST_LOOP; i++) {
                convertToRGB(d_inputV210, d_outputRGB8, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height, g_ctx.batch, lookupTable_cuda, stream);
            }
        } else if (c == 'n') {
            checkCudaErrors(cudaMalloc((void **)&d_outputRGB10, size * RGB_SIZE * sizeof(unsigned short)));
            cudaEventRecord(start, 0);
            for (int i = 0; i < TEST_LOOP; i++) {
                convertToRGB(d_inputV210, d_outputRGB10, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height, g_ctx.batch, stream);
            }
        }
    } else if (n == 3) {
        int nn = 3;
        s_cin.clear();
        cout << "Convert flow : 1. v210 -> rgb (8 bit) -> resize -> nvjpg\n"
             << "               2. v210 -> resize -> rgb (8 bit) -> nvjpg\n"
             << "               3. v210 -> resize (8 bit) -> nvjpeg ? (1, 2 , (default) 3) ";
        getline(cin, s_cin);
        if (!s_cin.empty()) {
            istringstream ss(s_cin);
            ss >> nn;
        }
        cudaEventRecord(start, 0);
        for (int i = 0; i < TEST_LOOP; i++) {
            encode_images(d_inputV210, argv, g_ctx.width, g_ctx.height, g_ctx.batch, g_ctx.dst_width, g_ctx.dst_height, lookupTable_cuda, nn, stream);
        }
    } else {
        cout << "Error number, bye bye.\n";
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout.precision(3);
    if (n == 1) {
        if (c == 'n') {
            cout << fixed << "  CUDA v210 to p210(" << g_ctx.width << "x" << g_ctx.height << " --> "
            << g_ctx.width << "x" << g_ctx.height << "), "
            << "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
            << " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch << " ms/frame\n";
            c = 'n';
            s_cin.clear();
            cout << "Write the dpx file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name : ";
                cin >> s;
                dpxHandler(d_outputYUV422, g_ctx.width, g_ctx.height, g_ctx.batch, s, YUV422_PLANAR_SIZE);
            }
            c = 'n';
            s_cin.clear();
            cout << "Write the row file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name: ";
                cin >> s;
                dumpYUV(d_outputYUV422, g_ctx.width, g_ctx.height, g_ctx.batch, s, YUV422_PLANAR_SIZE, stream);
            }
            checkCudaErrors(cudaFree(d_outputYUV422));
        } else if (c == 'y') {
            cout << fixed << "  CUDA v210 to p208(" << g_ctx.width << "x" << g_ctx.height << " --> "
            << g_ctx.width << "x" << g_ctx.height << "), "
            << "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
            << " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch << " ms/frame\n";
            c = 'n';
            s_cin.clear();
            cout << "Write the dpx file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name : ";
                cin >> s;
                dpxHandler(d_outputYUV422_1, g_ctx.width, g_ctx.height, g_ctx.batch, s, YUV422_PLANAR_SIZE);
            }
            c = 'n';
            s_cin.clear();
            cout << "Write the row file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name: ";
                cin >> s;
                dumpYUV(d_outputYUV422_1, g_ctx.width, g_ctx.height, g_ctx.batch, s, YUV422_PLANAR_SIZE, stream);
            }
            checkCudaErrors(cudaFree(d_outputYUV422_1));
        }
    } else if (n == 2) {
        if (c == 'y') { // 10 bit to 8 bit
            cout << fixed << "  CUDA v210 to RGB 8 bit(" << g_ctx.width << "x" << g_ctx.height << " --> "
                << g_ctx.width << "x" << g_ctx.height << "), "
                << "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
                << " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch << " ms/frame\n";
            c = 'n';
            s_cin.clear();
            cout << "Write the dpx file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name : ";
                cin >> s;
                dpxHandler(d_outputRGB8, g_ctx.width, g_ctx.height, g_ctx.batch, s, RGB_SIZE);
            }
            c = 'n';
            s_cin.clear();
            cout << "Write the row file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name : ";
                cin >> s;
                dumpYUV(d_outputRGB8, g_ctx.width, g_ctx.height, g_ctx.batch, s, RGB_SIZE, stream);
            }
            checkCudaErrors(cudaFree(d_outputRGB8));
        } else if (c == 'n') {
            cout << fixed << "  CUDA v210 to RGB 10 bit(" << g_ctx.width << "x" << g_ctx.height << " --> "
                << g_ctx.width << "x" << g_ctx.height << "), "
                << "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
                << " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch << " ms/frame\n";
            c = 'n';
            s_cin.clear();
            cout << "Write the dpx file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name : ";
                cin >> s;
                dpxHandler(d_outputRGB10, g_ctx.width, g_ctx.height, g_ctx.batch, s, RGB_SIZE);
            }
            c = 'n';
            s_cin.clear();
            cout << "Write the row file ? (y/N) ";
            getline(cin, s_cin);
            if (!s_cin.empty()) {
                istringstream ss(s_cin);
                ss >> c;
            }
            if (c == 'y') {
                string s;
                cout << "File name : ";
                cin >> s;
                dumpYUV(d_outputRGB10, g_ctx.width, g_ctx.height, g_ctx.batch, s, RGB_SIZE, stream);
            }
            checkCudaErrors(cudaFree(d_outputRGB10));
        }
    } else if (n == 3) {
        cout << fixed << "  CUDA v210 to nvJPEG(" << g_ctx.width << "x" << g_ctx.height << " --> "
        << g_ctx.dst_width << "x" << g_ctx.dst_height << "), "
        << "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
        << " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch << " ms/frame\n";
    }
    /* release resources */
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(lookupTable_cuda));
    free(lookupTable);
}

int convert(int argc, char* argv[]) {
    unsigned short *d_inputV210;
    unsigned char *p208;
    int size = 0;
    if (!isGPUEnable()) {
        return EXIT_FAILURE;
    }
    if (parseCmdLine(argc, argv) < 0) {
        return EXIT_FAILURE;
    }
    g_ctx.ctx_pitch = (g_ctx.width + 47) / 48 * 128 / 2;
    int ctx_alignment = 32;
    g_ctx.ctx_pitch += (g_ctx.ctx_pitch % ctx_alignment != 0) ? (ctx_alignment - g_ctx.ctx_pitch % ctx_alignment) : 0;
    size = g_ctx.ctx_pitch * g_ctx.height * g_ctx.batch;

    // Load v210 yuv data into d_inputV210.
#if USE_UVM_MEM
    checkCudaErrors(cudaMallocManaged((void **)&d_inputV210,
        (g_ctx.ctx_pitch * g_ctx.ctx_heights * g_ctx.batch), cudaMemAttachHost));
    cout << "\nUSE_UVM_MEM\n";
#else
    checkCudaErrors(cudaMalloc((void **)&d_inputV210, size * sizeof(unsigned short)));
#endif
    if (loadV210Frame(d_inputV210)) {
        cerr << "failed to load data!\n";
        return EXIT_FAILURE;
    }

    cout << "V210 to P210\n";
    //v210ToP010(d_inputV210, argv[2]);
    p208 = (unsigned char *)malloc(g_ctx.dst_width * g_ctx.dst_height * 2);
    int nJPEGSize = 0;
    converter(d_inputV210, p208, g_ctx.width, g_ctx.height, g_ctx.dst_width, g_ctx.dst_height, &nJPEGSize);
    ofstream output_file("r.jpg", ios::out | ios::binary);
    output_file.write((char *)p208, nJPEGSize);
    output_file.close();

    checkCudaErrors(cudaFree(d_inputV210));
    free(p208);
    return EXIT_SUCCESS;
}

//      call this method to do  ---> v210 -> resize -> nvJPEG
void converter(unsigned short *v210Src, unsigned char *p208Dst, int nSrcW, int nSrcH, int nDstw, int nDstH, int *nJPEGSize) {
    int rowByte = (nSrcW + 47) / 48 * 128 / 2;
    encode_params_t en_params;
    size_t length = 0;
    int *lookupTable, *lookupTable_cuda;
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    lookupTable = (int *)malloc(sizeof(int) * 1024);
    checkCudaErrors(cudaMalloc((void**)&lookupTable_cuda, sizeof(int) * 1024));
    lookupTableF(lookupTable);
    cudaMemcpy(lookupTable_cuda, lookupTable, sizeof(int) * 1024, cudaMemcpyHostToDevice);

    // initialize nvjpeg structures
    checkCudaErrors(nvjpegCreateSimple(&en_params.nv_handle));
    checkCudaErrors(nvjpegEncoderStateCreate(en_params.nv_handle, &en_params.nv_enc_state, stream));
    checkCudaErrors(nvjpegEncoderParamsCreate(en_params.nv_handle, &en_params.nv_enc_params, stream));

    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(en_params.nv_enc_params, NVJPEG_CSS_422, stream));

    checkCudaErrors(cudaMalloc((void **)&en_params.t_16, rowByte * nSrcH * sizeof(unsigned short)));
    en_params.nv_image.pitch[0] = nDstw * sizeof(unsigned char);
    en_params.nv_image.pitch[1] = nDstw / 2 * sizeof(unsigned char);
    en_params.nv_image.pitch[2] = nDstw / 2 * sizeof(unsigned char);
    checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[0], nDstw * nDstH * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[1], nDstw * nDstH / 2 * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[2], nDstw * nDstH / 2 * sizeof(unsigned char)));

    checkCudaErrors(cudaMemcpy((void *)en_params.t_16, (void *)v210Src, rowByte * nSrcH * sizeof(unsigned short), cudaMemcpyHostToDevice));
    resizeBatch(en_params.t_16, rowByte, nSrcH,
        en_params.nv_image.channel[0], en_params.nv_image.channel[1], en_params.nv_image.channel[2],
        nDstw, nDstH, 1, lookupTable_cuda, stream);

    checkCudaErrors(nvjpegEncodeYUV(en_params.nv_handle, en_params.nv_enc_state, en_params.nv_enc_params,
        &en_params.nv_image, NVJPEG_CSS_422, nDstw, nDstH, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));
    // get compressed stream size
    checkCudaErrors(
      nvjpegEncodeRetrieveBitstream(en_params.nv_handle, en_params.nv_enc_state, NULL, &length, stream));

    // get stream itself
    checkCudaErrors(cudaStreamSynchronize(stream));
    vector<unsigned char> jpeg(length);
    checkCudaErrors(nvjpegEncodeRetrieveBitstream(en_params.nv_handle, en_params.nv_enc_state, jpeg.data(), &length, stream));

    // write stream to file
    checkCudaErrors(cudaStreamSynchronize(stream));
    memcpy(p208Dst, jpeg.data(), length);
    *nJPEGSize = length;
    checkCudaErrors(cudaFree(en_params.t_16));
    checkCudaErrors(cudaFree(en_params.nv_image.channel[0]));
    checkCudaErrors(cudaFree(en_params.nv_image.channel[1]));
    checkCudaErrors(cudaFree(en_params.nv_image.channel[2]));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(lookupTable_cuda));
    free(lookupTable);
    checkCudaErrors(nvjpegDestroy(en_params.nv_handle));
    checkCudaErrors(nvjpegEncoderStateDestroy(en_params.nv_enc_state));
    checkCudaErrors(nvjpegEncoderParamsDestroy(en_params.nv_enc_params));
}
