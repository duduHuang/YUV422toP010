#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <iomanip>
#include <csignal>

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "DuDuConvertAPI.h"
#include "DuDuRGBConvertAPI.h"

std::condition_variable g_sleepCond;
std::mutex g_sleepMutex;
volatile std::sig_atomic_t g_doExit = 0;

static void sigfunc(int signum)
{
    std::cout << "Interrupt signal(" << signum << ") received." << std::endl;
    if (signum == SIGINT || signum == SIGTERM)
    {
        g_doExit = 1;
        g_sleepCond.notify_all();
    }
}

int main(int argc, char *argv[])
{    
    signal(SIGINT, sigfunc);
    signal(SIGTERM, sigfunc);

    int32_t srcWidth = 7680;
    int32_t srcHeight = 4320;
    int64_t srcRowBytes = (7680 + 47)/48*128;

    int32_t dstWidth = 1280;
    int32_t dstHeight = 720;

    uint8_t* srcBuffer = new uint8_t[srcRowBytes*srcHeight];
    std::cout << "Src Buffer Size: " << srcRowBytes*srcHeight << std::endl;;

    uint8_t* dstBuffer = new uint8_t[dstWidth*dstHeight*2];
    std::cout << "Dst Buffer Size: " << dstWidth*dstHeight*2 << std::endl;

    uint64_t frameIdx = 0;

    IDuDuConverter*  m_duduConverter;
    m_duduConverter = DuDuConverterAPICreate();

    if (!m_duduConverter)
    {
    	std::cout << "Cannot create dudu converter";
    	return 1;
    }

    if (!m_duduConverter->IsGPUSupport())
    {
		std::cout << "There is no GPU card.";
		return 1;
    }

    m_duduConverter->Initialize();

    m_duduConverter->SetSrcSize(srcWidth, srcHeight);
    m_duduConverter->SetDstSize(dstWidth, dstHeight);
    m_duduConverter->AllocateMem();

    std::string saveLocation = "/tmp/ramdisk/preview.jpg";

    std::chrono::high_resolution_clock::time_point tpInvlStart = std::chrono::high_resolution_clock::now();
    std::chrono::steady_clock::time_point tpStart;
    uint64_t frameCompleteCount = 0;
    float fps = 59.94f;
    uint64_t timeInvl = uint64_t(1000.f / fps * 1000.f);
    std::cout << "Time Interval: " << timeInvl << std::endl;
    while (!g_doExit)
    {
		std::this_thread::yield();
		std::chrono::high_resolution_clock::time_point tpInvlEnd = std::chrono::high_resolution_clock::now();

        uint64_t diffMicrosecond = std::chrono::duration_cast<std::chrono::microseconds>(tpInvlEnd - tpInvlStart).count();
    	if (diffMicrosecond > timeInvl)
        {
        	if (frameCompleteCount == 0)
        		tpStart = std::chrono::steady_clock::now();

        	int32_t jpgSize = 0;
            m_duduConverter->ConvertAndResize((uint16_t*)srcBuffer, dstBuffer, &jpgSize);

            if (jpgSize > 0)
            {
            	std::ofstream outputFile(saveLocation, std::ios::out | std::ios::binary);

	            if (!outputFile.good())
	            {
	                std::cout << "Cannot write jpg file." << std::endl;
	            }

	            outputFile.write((char*)dstBuffer, jpgSize);
	            outputFile.close();
            }

	    	frameIdx++;
	    	frameCompleteCount++;
	    	tpInvlStart = tpInvlEnd;

	    	if (frameCompleteCount == 120)
	    	{
	    	    std::chrono::steady_clock::time_point tpEnd = std::chrono::steady_clock::now();
	    		int64_t frmDiffMs = std::chrono::duration_cast<std::chrono::milliseconds>(tpEnd - tpStart).count();
	    		float fps = (float)frameCompleteCount / (float)frmDiffMs * 1000.f;

	    		std::stringstream ss;
	        	ss << "FPS: " << fps << ", " << frmDiffMs << "ms" << std::endl;
	        	std::cout << ss.str() << std::endl;
	        	frameCompleteCount = 0;
	    	}
    	}
    }

  	m_duduConverter->FreeMemory();
    m_duduConverter->Destroy();
    delete srcBuffer;
    delete dstBuffer;

    return 0;
}