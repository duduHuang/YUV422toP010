#include "convert.h"
#include "convertToP010.h"
#include "convertToRGB.h"

nv210_to_p010_context_t g_ctx;

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
	//	d_V210 += frameSize;
	//}
#if (USE_UVM_MEM == 0)
	free(pV210FrameData);
#endif
	v210File.close();
	return EXIT_SUCCESS;
}

/*
  Draw yuv data
*/
void dumpYUV(unsigned short *d_srcNv12, int width, int height, int batch, char *filename) {
	unsigned short *nv12Data, *d_nv12;
	int size = 0;
	size = g_ctx.ctx_pitch * g_ctx.height * 2;

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

/*
  Convert v210 to p010
*/
void v210ToP010(unsigned short *d_inputV210, char *argv) {
	unsigned short *d_outputYUV422;
	int size = 0, block_size = 0;

	size = g_ctx.ctx_pitch * g_ctx.height * g_ctx.batch;
	checkCudaErrors(cudaMalloc((void **)&d_outputYUV422, size * 2 * sizeof(unsigned short)));
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
		convertToP010(d_inputV210, g_ctx.ctx_pitch, d_outputYUV422,
			g_ctx.ctx_pitch, g_ctx.height, g_ctx.batch, block_size, 0);
		//convertToRGB(d_inputV210, g_ctx.ctx_pitch, d_outputYUV422,
		//	g_ctx.ctx_pitch, g_ctx.height, g_ctx.batch, block_size, 0);
	}
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));

	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	cout.precision(3);
	cout << fixed << "  CUDA v210 to p210(" << g_ctx.width << "x" << g_ctx.height << " --> "
		<< g_ctx.width << "x" << g_ctx.height << "), "
		<< "average time: " << (elapsedTime / (TEST_LOOP * 1.0f)) << "ms"
		<< " ==> " << (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch << " ms/frame\n";

	dumpYUV(d_outputYUV422, g_ctx.ctx_pitch, g_ctx.height, g_ctx.batch, argv);

	/* release resources */
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaStreamDestroy(stream));
	checkCudaErrors(cudaFree(d_outputYUV422));
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
	v210ToP010(d_inputV210, argv[1]);

	checkCudaErrors(cudaFree(d_inputV210));
	return EXIT_SUCCESS;
}
