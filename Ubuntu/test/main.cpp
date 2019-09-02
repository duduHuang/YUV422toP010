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

#include "DuDuGPUSupportAPI.h"
#include "DuDuV210ConvertAPI.h"
#include "DuDuRGBConvertAPI.h"

using namespace std;

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

void testDuDuConvert() {
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

    IDuDuV210Converter*  m_duduConverter;
    m_duduConverter = DuDuV210ConverterAPICreate();

    if (!m_duduConverter)
    {
        std::cout << "Cannot create dudu converter";
        return;
    }

    m_duduConverter->Initialize();
    m_duduConverter->SetSrcSize(srcWidth, srcHeight);
    m_duduConverter->SetDstSize(dstWidth, dstHeight);
    m_duduConverter->AllocateMem();

    std::string saveLocation = "../v210.yuv";

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
    delete srcBuffer;
    delete dstBuffer;
}

void testRGBConvert() {
    signal(SIGINT, sigfunc);
    signal(SIGTERM, sigfunc);

    unsigned short *dSrc;
    int frameSize = 7680 * 4320 * 3;
    dSrc = new unsigned short[frameSize * sizeof(unsigned short)];
    ifstream iFile("../../rgb10.rgb", ifstream::in | ios::binary);
    if (!iFile.is_open()) {
        cerr << "Can't open files\n";
        return;
    }
    iFile.read((char *)dSrc, frameSize * sizeof(unsigned short));
    if (iFile.gcount() < frameSize) {
        cerr << "can't get one frame\n";
        return;
    }
    iFile.close();

    frameSize = (7680 + 47) / 48 * 128 * 4320;
    unsigned short *v210 = new unsigned short[frameSize * sizeof(unsigned short)];
    unsigned char *rgb = new unsigned char[1280 * 720 * 3];

    IDuDuRGBConverter *converterTool;
    converterTool = DuDuRGBConverterAPICreate();

    converterTool->Initialize();
	converterTool->SetSrcSize(7680, 4320);
	converterTool->SetDstSize(1280, 720);

	converterTool->AllocateSrcAndTableMem();
	converterTool->SetCudaDevSrc(dSrc);
	converterTool->AllocateV210DstMem();
	converterTool->AllocatNVJPEGRGBMem();

    std::chrono::high_resolution_clock::time_point tpInvlStart = std::chrono::high_resolution_clock::now();
    std::chrono::steady_clock::time_point tpStart;
    uint64_t frameCompleteCount = 0;
    float fps = 59.94f;
    uint64_t timeInvl = uint64_t(1000.f / fps * 1000.f);
    std::cout << "Time Interval: " << timeInvl << std::endl;

    uint64_t frameIdx = 0;
    while (!g_doExit) {
        std::this_thread::yield();
        std::chrono::high_resolution_clock::time_point tpInvlEnd = std::chrono::high_resolution_clock::now();

        uint64_t diffMicrosecond = std::chrono::duration_cast<std::chrono::microseconds>(tpInvlEnd - tpInvlStart).count();
        if (diffMicrosecond > timeInvl) {
            if (frameCompleteCount == 0) {
                tpStart = std::chrono::steady_clock::now();
            }

            int nJPEGSize = 0;
            converterTool->RGB10ConvertAndResizeToNVJPEG(rgb, &nJPEGSize);
            converterTool->RGB10ConvertToV210(v210);

            if (nJPEGSize > 0) {
                std::ofstream outputFile("rgb8.jpg", std::ios::out | std::ios::binary);

                if (!outputFile.good()) {
                    std::cout << "Cannot write jpg file." << std::endl;
                }

                outputFile.write((char*)rgb, nJPEGSize);
                outputFile.close();

                std::ofstream oFile("tV210.yuv", std::ios::out | std::ios::binary);

                if (!oFile.good()) {
                    std::cout << "Cannot write jpg file." << std::endl;
                }

                oFile.write((char*)v210, frameSize);
                oFile.close();
            }

            frameIdx++;
            frameCompleteCount++;
            tpInvlStart = tpInvlEnd;

            if (frameCompleteCount == 120) {
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

    delete[] v210;
    converterTool->FreeMemory();
    delete[] dSrc;
    delete[] rgb;
}

void test() {
    IDuDuRGBConverter *converterToolRGB;
    converterToolRGB = DuDuRGBConverterAPICreate();

    IDuDuV210Converter*  m_duduConverter;
    m_duduConverter = DuDuV210ConverterAPICreate();

    int32_t srcWidth = 7680;
    int32_t srcHeight = 4320;
    int64_t srcRowBytes = (7680 + 47)/48*128;

    int32_t dstWidth = 1280;
    int32_t dstHeight = 720;

	uint8_t* srcBuffer = new uint8_t[srcRowBytes*srcHeight];
    std::cout << "Src Buffer Size: " << srcRowBytes*srcHeight << std::endl;;

    uint8_t* dstBuffer = new uint8_t[dstWidth*dstHeight*2];
    std::cout << "Dst Buffer Size: " << dstWidth*dstHeight*2 << std::endl;

    unsigned short *dSrc;
    int frameSize = 7680 * 4320 * 3;
    dSrc = new unsigned short[frameSize * sizeof(unsigned short)];
    ifstream iFile("../../rgb10.rgb", ifstream::in | ios::binary);
    if (!iFile.is_open()) {
        cerr << "Can't open files\n";
        return;
    }
    iFile.read((char *)dSrc, frameSize * sizeof(unsigned short));
    if (iFile.gcount() < frameSize) {
        cerr << "can't get one frame\n";
        return;
    }
    iFile.close();

	frameSize = (7680 + 47) / 48 * 128 * 4320;
    unsigned short *v210 = new unsigned short[frameSize * sizeof(unsigned short)];
    unsigned char *rgb = new unsigned char[1280 * 720 * 3];

    if (!IsGPUSupport()) {
        std::cout << "There is no GPU card.";
        return;
    }

    m_duduConverter->Initialize();
    m_duduConverter->SetSrcSize(srcWidth, srcHeight);
    m_duduConverter->SetDstSize(dstWidth, dstHeight);
    m_duduConverter->AllocateMem();

    converterToolRGB->Initialize();
	converterToolRGB->SetSrcSize(srcWidth, srcHeight);
    converterToolRGB->SetDstSize(dstWidth, dstHeight);
    converterToolRGB->AllocateSrcAndTableMem();
    converterToolRGB->SetCudaDevSrc(dSrc);
    converterToolRGB->AllocateV210DstMem();
    converterToolRGB->AllocatNVJPEGRGBMem();

	std::chrono::high_resolution_clock::time_point tpInvlStart = std::chrono::high_resolution_clock::now();
    std::chrono::steady_clock::time_point tpStart;
    uint64_t frameCompleteCount = 0;
    float fps = 59.94f;
    uint64_t timeInvl = uint64_t(1000.f / fps * 1000.f);
    std::cout << "Time Interval: " << timeInvl << std::endl;

	uint64_t frameIdx = 0;
    while (!g_doExit) {
        std::this_thread::yield();
        std::chrono::high_resolution_clock::time_point tpInvlEnd = std::chrono::high_resolution_clock::now();

        uint64_t diffMicrosecond = std::chrono::duration_cast<std::chrono::microseconds>(tpInvlEnd - tpInvlStart).count();
        if (diffMicrosecond > timeInvl) {
            if (frameCompleteCount == 0) {
                tpStart = std::chrono::steady_clock::now();
            }

            int nJPEGSize = 0;
			int32_t jpgSize = 0;
			m_duduConverter->ConvertAndResize((uint16_t*)srcBuffer, dstBuffer, &jpgSize);
            converterToolRGB->RGB10ConvertAndResizeToNVJPEG(rgb, &nJPEGSize);
            converterToolRGB->RGB10ConvertToV210(v210);

            if (nJPEGSize > 0) {
                std::ofstream outputFile("rgb8.jpg", std::ios::out | std::ios::binary);

                if (!outputFile.good()) {
                    std::cout << "Cannot write jpg file." << std::endl;
                }

                outputFile.write((char*)rgb, nJPEGSize);
                outputFile.close();

                std::ofstream oFile("tV210.yuv", std::ios::out | std::ios::binary);

                if (!oFile.good()) {
                    std::cout << "Cannot write jpg file." << std::endl;
                }

                oFile.write((char*)v210, frameSize);
                oFile.close();
            }

            frameIdx++;
            frameCompleteCount++;
            tpInvlStart = tpInvlEnd;

            if (frameCompleteCount == 120) {
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
    converterToolRGB->FreeMemory();
	delete srcBuffer;
    delete dstBuffer;
    delete[] dSrc;
	delete[] v210;
	delete[] rgb;
}

int main(int argc, char *argv[]) {
    //testDuDuConvert();
    //testRGBConvert();
    test();
    return 0;
}