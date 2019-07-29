#include "multiStream.h"
#include "convertToP210.h"
#include "convertToRGB.h"
#include "libdpx/DPX.h"
#include "resize.h"

using namespace dpx;

int parseCmdLine(int argc, char *argv[], nv210_context_t *g_ctx) {
	int w = 7680, h = 4320;
	string s;
	//if (argc == 3) {
		// Run using default arguments
		g_ctx->input_v210_file = "v210.yuv";
		if (g_ctx->input_v210_file == NULL) {
			cout << "Cannot find input file" << argv[1] << "\n Exiting\n";
			return -1;
		}
		g_ctx->width = 7680;
		g_ctx->height = 4320;
		g_ctx->batch = 1;
		cout << "Output resolution: (7680 4320)\n"
			<< "                   (3840 2160)\n"
			<< "                   (1920 1080)\n"
			<< "                   default (1280 720) ";
		getline(cin, s);
		if (!s.empty()) {
			istringstream ss(s);
			ss >> w >> h;
		}
		g_ctx->dst_width = w;
		g_ctx->dst_height = h;
	//}
	g_ctx->device = findCudaDevice(argc, (const char **)argv);
	if (g_ctx->width == 0 || g_ctx->height == 0 || !g_ctx->input_v210_file) {
		cout << "Usage: " << argv[0] << " inputf outputf\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

/*
  Load v210 yuvfile data into host memory
*/
static int loadV210Frame(unsigned short *dSrc, nv210_context_t *g_ctx) {
	int frameSize = g_ctx->width * g_ctx->height * 4 / 3;
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
	if (bPinGenericMemory) 	{
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

/*
  Convert v210 to other type
*/
void multiStream(unsigned short *d_src, char *argv, nv210_context_t *g_ctx, int device_sync_method, bool bPinGenericMemory) {
	// create CUDA event handles
	// use blocking sync
	cudaEvent_t start_event, stop_event;
	unsigned short *dV210Device, *dP210, *dAligned_P210, *dP210Device;
	const int nStreams = 4;
	int v210Size = 0, byteChunkSize = 0, byteOffset = 0, p210Size = 0, p210ByteChunkSize = 0, p210ByteOffset = 0;
	float elapsed_time = 0.0f;
	cout << " \n";
	cout << "Computing results using GPU, using " << nStreams << " streams.\n";
	cout << " \n";
	// allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));
	cout << "    Creating " << nStreams << " CUDA streams.\n";
	for (int i = 0; i < nStreams; i++) {
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}
	int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync : cudaEventDefault);
	checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
	checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));
	v210Size = g_ctx->width * g_ctx->height * 4 / 3;
	p210Size = g_ctx->width * g_ctx->height * YUV422_PLANAR_SIZE;
	byteChunkSize = (v210Size * sizeof(unsigned short)) / nStreams;
	p210ByteChunkSize = (p210Size * sizeof(unsigned short)) / nStreams;

	// Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
	AllocateHostMemory(bPinGenericMemory, &dP210, &dAligned_P210, p210Size * sizeof(unsigned short));
	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&dV210Device, v210Size * sizeof(unsigned short))); // pointers to data in the device memory
	checkCudaErrors(cudaMalloc((void **)&dP210Device, p210Size * sizeof(unsigned short))); // pointers to data in the device memory
	checkCudaErrors(cudaEventRecord(start_event, 0));
	checkCudaErrors(cudaMemcpyAsync(dV210Device, d_src, v210Size * sizeof(unsigned short), cudaMemcpyHostToDevice, streams[0]));
	convertToP210(dV210Device, dP210Device, g_ctx->width, g_ctx->height, streams[0]);
	checkCudaErrors(cudaMemcpyAsync(dP210, dP210Device, p210Size * sizeof(unsigned short)	, cudaMemcpyDeviceToHost, streams[0]));
	checkCudaErrors(cudaEventRecord(stop_event, 0));
	checkCudaErrors(cudaEventSynchronize(stop_event));
	checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	cout << fixed << "  CUDA v210 to p210(" << g_ctx->width << "x" << g_ctx->height << " --> "
		<< g_ctx->width << "x" << g_ctx->height << "), "
		<< "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << "ms"
		<< " ==> " << (elapsed_time / (TEST_LOOP * 1.0f)) / g_ctx->batch << " ms/frame\n";
	checkCudaErrors(cudaEventRecord(start_event, 0));
	for (int i = 0; i < nStreams; i++) {
	    cout << "        Launching stream " << i << ".\n";
		byteOffset = v210Size * i / nStreams;
		p210ByteOffset = p210Size * i / nStreams;
		checkCudaErrors(cudaMemcpyAsync(dV210Device + byteOffset, d_src + byteOffset,
			byteChunkSize, cudaMemcpyHostToDevice, streams[i]));
		convertToP210(dV210Device + byteOffset, dP210Device + p210ByteOffset,
			g_ctx->width, g_ctx->height / nStreams, streams[i]);
		checkCudaErrors(cudaMemcpyAsync(dP210 + p210ByteOffset, dP210Device + p210ByteOffset,
			p210ByteChunkSize, cudaMemcpyDeviceToHost, streams[i]));
	}
	checkCudaErrors(cudaEventRecord(stop_event, 0));
	checkCudaErrors(cudaEventSynchronize(stop_event));
	checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	cout << fixed << "  CUDA v210 to p210(" << g_ctx->width << "x" << g_ctx->height << " --> "
		<< g_ctx->width << "x" << g_ctx->height << "), "
		<< "average time: " << (elapsed_time / (TEST_LOOP * 1.0f)) << "ms"
		<< " ==> " << (elapsed_time / (TEST_LOOP * 1.0f)) / g_ctx->batch << " ms/frame\n";
	// release resources
	// Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
	FreeHostMemory(bPinGenericMemory, &dP210, &dAligned_P210, p210Size * sizeof(unsigned short));
	checkCudaErrors(cudaFree(dV210Device));
	checkCudaErrors(cudaFree(dP210Device));
	for (int i = 0; i < nStreams; i++)
	{
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}
	checkCudaErrors(cudaEventDestroy(start_event));
	checkCudaErrors(cudaEventDestroy(stop_event));
}

int multiStream(int argc, char* argv[]) {
	nv210_context_t *g_ctx;
	// allocate generic memory and pin it laster instead of using cudaHostAlloc()
	bool bPinGenericMemory = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
	int  device_sync_method = cudaDeviceScheduleAuto; // by default we use BlockingSync
	checkCudaErrors(cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0)));
	unsigned short *srcV210, *srcAligned_V210;
	int v210Size = 0;
	g_ctx = (nv210_context_t *)malloc(sizeof(nv210_context_t));
	if (parseCmdLine(argc, argv, g_ctx) < 0) {
		return EXIT_FAILURE;
	}
	v210Size = g_ctx->width * g_ctx->height * 4 / 3;
	// Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
	AllocateHostMemory(bPinGenericMemory, &srcV210, &srcAligned_V210, v210Size * sizeof(unsigned short));
	// Load v210 yuv data into d_inputV210.
	if (loadV210Frame(srcV210, g_ctx)) {
		cerr << "failed to load data!\n";
		return EXIT_FAILURE;
	}
	cout << "V210 to P210\n";
	multiStream(srcV210, argv[2], g_ctx, device_sync_method, bPinGenericMemory);
	FreeHostMemory(bPinGenericMemory, &srcV210, &srcAligned_V210, v210Size * sizeof(unsigned short));
	return EXIT_SUCCESS;
}