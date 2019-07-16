#include "convert.h"
#include "convertToP010.h"
#include "convertToRGB.h"
#include "libdpx/DPX.h"
#include "resize.h"

using namespace dpx;

nv210_to_p010_context_t g_ctx;
encode_params_t en_params;

int parseCmdLine(int argc, char *argv[]) {

    memset(&g_ctx, 0, sizeof(g_ctx));

    if (argc == 3) {
        // Run using default arguments
        g_ctx.input_v210_file = sdkFindFilePath(argv[1], argv[0]);
        if (g_ctx.input_v210_file == NULL) {
            cout << "Cannot find input file v210_000.yuv\n Exiting\n";
            return -1;
        }
        g_ctx.width = 7680;
        g_ctx.height = 4320;
        g_ctx.batch = 1;
        g_ctx.dst_width = 1920;
        g_ctx.dst_height = 1080;
    }

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

    frameSize = g_ctx.ctx_pitch * g_ctx.height * 4 / 3;

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

/*
  DPX header
*/
int dpxHandler(unsigned short *src) {
    unsigned short *nv12Data, *d_nv12;
    int size = 0;
    OutStream img;
    Writer writerf;

    size = g_ctx.ctx_pitch * g_ctx.height * RGB_SIZE;

    nv12Data = (unsigned short *)malloc(size * sizeof(unsigned short));
    if (nv12Data == NULL) {
        cerr << "Failed to allcoate memory\n";
        return EXIT_FAILURE;
    }
    memset((void *)nv12Data, 0, size * sizeof(unsigned short));
    d_nv12 = src;
    cudaMemcpy((void *)nv12Data, (void *)d_nv12, size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    char fileName[] = "ddppxx.dpx";

    if (!img.Open(fileName)) {
        cout << "Unable to open file ddppxx.dpx" << endl;
        return EXIT_FAILURE;
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
    writerf.header.SetPixelsPerLine(g_ctx.ctx_pitch);
    writerf.WriteHeader();
    writerf.WriteElement(0, (void *)nv12Data);
    writerf.Finish();

    img.Close();
    free(nv12Data);
    return EXIT_SUCCESS;
}

/*
  Jpeg encoder
*/
int encode_images(unsigned short *d_inputV210, int width, int height, int batch, int dstWidth, int dstHeight, cudaStream_t stream) {

    size_t length = 0;

    // initialize nvjpeg structures
    checkCudaErrors(nvjpegCreateSimple(&en_params.nv_handle));
    checkCudaErrors(nvjpegEncoderStateCreate(en_params.nv_handle, &en_params.nv_enc_state, stream));
    checkCudaErrors(nvjpegEncoderParamsCreate(en_params.nv_handle, &en_params.nv_enc_params, stream));

    // Set encode parameters
    //checkCudaErrors(nvjpegEncoderParamsSetQuality(en_params.nv_enc_params, 70, stream));
    //checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(en_params.nv_enc_params, 0, stream));
    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(en_params.nv_enc_params, NVJPEG_CSS_444, stream));
    en_params.nv_image.pitch[0] = dstWidth * RGB_SIZE * sizeof(unsigned char);
    checkCudaErrors(cudaMalloc(&en_params.t, width * height * RGB_SIZE * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&en_params.nv_image.channel[0], dstWidth * dstHeight * RGB_SIZE * sizeof(unsigned char)));

    convertToRGB(d_inputV210, width, en_params.t, width, height, batch, 0, stream);
    resizeBatch(en_params.t, width, height, en_params.nv_image.channel[0], dstWidth, dstHeight, batch, stream);

    en_params.nv_image.pitch[0] = dstWidth * RGB_SIZE * sizeof(unsigned char);
    // Compress image
    checkCudaErrors(nvjpegEncodeImage(en_params.nv_handle, en_params.nv_enc_state, en_params.nv_enc_params,
        &en_params.nv_image, NVJPEG_INPUT_RGBI, dstWidth, dstHeight, stream));
    //checkCudaErrors(nvjpegEncodeYUV(en_params.nv_handle, en_params.nv_enc_state, en_params.nv_enc_params,
    //    &en_params.nv_image, NVJPEG_CSS_422, width, height, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    // get compressed stream size

    checkCudaErrors(
      nvjpegEncodeRetrieveBitstream(en_params.nv_handle, en_params.nv_enc_state, NULL, &length, stream));

    // get stream itself
    checkCudaErrors(cudaStreamSynchronize(stream));
    vector<unsigned char> jpeg(length);
    checkCudaErrors(
    nvjpegEncodeRetrieveBitstream(en_params.nv_handle, en_params.nv_enc_state, jpeg.data(), &length, stream));

    // write stream to file
    checkCudaErrors(cudaStreamSynchronize(stream));
    ofstream output_file("test.jpg", ios::out | ios::binary);
    output_file.write((char *)jpeg.data(), length);
    output_file.close();
    checkCudaErrors(cudaFree(en_params.nv_image.channel[0]));
    checkCudaErrors(nvjpegDestroy(en_params.nv_handle));
    checkCudaErrors(nvjpegEncoderStateDestroy(en_params.nv_enc_state));
    checkCudaErrors(nvjpegEncoderParamsDestroy(en_params.nv_enc_params));
    return EXIT_SUCCESS;
}

/*
  Draw yuv data
*/
void dumpYUV(unsigned short *d_srcNv12, int width, int height, int batch, char *filename, cudaStream_t stream) {
    unsigned short *nv12Data, *d_nv12;
    int size = 0;
    size = g_ctx.ctx_pitch * g_ctx.height * RGB_SIZE;

    //nv12Data = (unsigned short *)malloc(size * sizeof(unsigned short));
    //if (nv12Data == NULL) {
    //    cerr << "Failed to allcoate memory\n";
    //    return;
    //}
    //memset((void *)nv12Data, 0, size * sizeof(unsigned short));
    d_nv12 = d_srcNv12;
    for (int i = 0; i < batch; i++) {
        //ofstream nv12File(filename, ostream::out | ostream::binary);

        //cudaMemcpy((void *)nv12Data, (void *)d_nv12, size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        //if (nv12File) {
        //    nv12File.write((char *)nv12Data, size * sizeof(unsigned short));
        //    nv12File.close();
        //}
        d_nv12 += width * height;
    }
    //free(nv12Data);
}

void dumpYUV(unsigned char *d_srcNv12, int width, int height, int batch, char *filename, cudaStream_t stream) {
    unsigned char *nv12Data, *d_nv12;
    int size = 0;
    size = g_ctx.ctx_pitch * g_ctx.height * RGB_SIZE;

    nv12Data = (unsigned char *)malloc(size * sizeof(unsigned short));
    if (nv12Data == NULL) {
        cerr << "Failed to allcoate memory\n";
        return;
    }
    memset((void *)nv12Data, 0, size * sizeof(unsigned short));
    d_nv12 = d_srcNv12;
    for (int i = 0; i < batch; i++) {
        //ofstream nv12File(filename, ostream::out | ostream::binary);

        //cudaMemcpy((void *)nv12Data, (void *)d_nv12, size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        //if (nv12File) {
        //    nv12File.write((char *)nv12Data, size * sizeof(unsigned short));
        //    nv12File.close();
        //}
        d_nv12 += width * height;
    }
    free(nv12Data);
}

/*
  Convert v210 to p010
*/
void v210ToP010(unsigned short *d_inputV210, char *argv) {
    //unsigned short *d_outputYUV422;
    int size = 0, block_size = 0;

    size = g_ctx.ctx_pitch * g_ctx.height * g_ctx.batch;
    //checkCudaErrors(cudaMalloc((void **)&d_outputYUV422, size * RGB_SIZE * sizeof(unsigned short)));
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsedTime = 0.0f;

    // convert to p010
    cudaEventRecord(start, 0);
    for (int i = 0; i < TEST_LOOP; i++) {
        //convertToP010(d_inputV210, g_ctx.ctx_pitch, d_outputYUV422,
        //  g_ctx.ctx_pitch, g_ctx.height, g_ctx.batch, block_size, 0);
        encode_images(d_inputV210, g_ctx.ctx_pitch, g_ctx.height, g_ctx.batch, g_ctx.dst_width, g_ctx.dst_height, stream);
        //convertToRGB(d_inputV210, g_ctx.ctx_pitch, d_outputYUV422,
        //    g_ctx.ctx_pitch, g_ctx.height, g_ctx.batch, block_size, 0);
        //dumpYUV(d_outputYUV422, g_ctx.ctx_pitch, g_ctx.height, g_ctx.batch, argv, stream);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout.precision(3);
    cout << fixed << "  CUDA v210 to p210(" << g_ctx.width << "x" << g_ctx.height << " --> "
        << g_ctx.width << "x" << g_ctx.height << "), "
        << "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
        << " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch << " ms/frame\n";

    //dpxHandler(d_outputYUV422);

    /* release resources */
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
    //checkCudaErrors(cudaFree(d_outputYUV422));
}

int convert(int argc, char* argv[]) {
    unsigned short *d_inputV210;
    int size = 0;
    if (parseCmdLine(argc, argv) < 0) {
        return EXIT_FAILURE;
    }
    g_ctx.ctx_pitch = g_ctx.width;
    int ctx_alignment = 32;
    g_ctx.ctx_pitch += (g_ctx.ctx_pitch % ctx_alignment != 0) ? (ctx_alignment - g_ctx.ctx_pitch % ctx_alignment) : 0;
    size = g_ctx.ctx_pitch * g_ctx.height * 4 / 3 * g_ctx.batch;

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
    v210ToP010(d_inputV210, argv[2]);

    checkCudaErrors(cudaFree(d_inputV210));
    return EXIT_SUCCESS;
}
