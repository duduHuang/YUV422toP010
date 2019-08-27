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
#include "convertTool.h"

using namespace std;

std::condition_variable g_sleepCond;
std::mutex g_sleepMutex;
volatile std::sig_atomic_t g_doExit = 0;

void testConvertToP208();
void testRGBConvert();

static void sigfunc(int signum)
{
    std::cout << "Interrupt signal(" << signum << ") received." << std::endl;
    if (signum == SIGINT || signum == SIGTERM)
    {
        g_doExit = 1;
        g_sleepCond.notify_all();
    }
}

void testConvertToP208() {
    signal(SIGINT, sigfunc);
    signal(SIGTERM, sigfunc);

    unsigned short *dSrc;
    int frameSize = (7680 + 47) / 48 * 128 * 4320;
    dSrc = new unsigned short[frameSize];
    ifstream v210File("v210.yuv", ifstream::in | ios::binary);
    if (!v210File.is_open()) {
        cerr << "Can't open files\n";
        return;
    }
    v210File.read((char *)dSrc, frameSize);
    if (v210File.gcount() < frameSize) {
        cerr << "can't get one frame\n";
        return;
    }
    v210File.close();

    unsigned char *p208;
    p208 = new unsigned char[1280 * 720 * 2];

    ConverterTool *converterTool;
    converterTool = new ConverterTool();

    if (converterTool->isGPUEnable()) {
        converterTool->initialCuda();
        converterTool->lookupTableF();
        converterTool->setSrcSize(7680, 4320);
        converterTool->setDstSize(1280, 720);
        converterTool->allocateMem();
    }
    else {
        cout << "device hasn't cuda !!!\n";
        delete[] p208;
        converterTool->freeMemory();
        converterTool->destroyCudaEvent();
        delete[] dSrc;
        return;
    }

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
            converterTool->convertToP208ThenResize(dSrc, p208, &nJPEGSize);

            if (nJPEGSize > 0) {
                std::ofstream outputFile("p208.jpg", std::ios::out | std::ios::binary);

                if (!outputFile.good()) {
                    std::cout << "Cannot write jpg file." << std::endl;
                }

                outputFile.write((char*)p208, nJPEGSize);
                outputFile.close();
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

    delete[] p208;
    converterTool->freeMemory();
    converterTool->destroyCudaEvent();
    delete[] dSrc;
}

void testRGBConvert() {
    signal(SIGINT, sigfunc);
    signal(SIGTERM, sigfunc);

    unsigned short *dSrc;
    int frameSize = 7680 * 4320 * 3;
    dSrc = new unsigned short[frameSize * sizeof(unsigned short)];
    ifstream iFile("rgb10.rgb", ifstream::in | ios::binary);
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

    ConverterTool *converterTool;
    converterTool = new ConverterTool();

    if (converterTool->isGPUEnable()) {
        converterTool->initialCuda();
        converterTool->lookupTableF();
        converterTool->setSrcSize(7680, 4320);
        converterTool->setDstSize(1280, 720);

        converterTool->allocatSrcMem();
        converterTool->setCudaDevSrc(dSrc);
        converterTool->allocatNVJPEGRGBMem();
        converterTool->allocatV210DstMem();
    }
    else {
        cout << "device hasn't cuda !!!\n";
        delete[] v210;
        converterTool->freeMemory();
        converterTool->destroyCudaEvent();
        delete[] dSrc;
        delete[] rgb;
        return;
    }

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
            converterTool->RGB10ConvertToRGB8NVJPEG(rgb, &nJPEGSize);
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
    converterTool->freeMemory();
    converterTool->destroyCudaEvent();
    delete[] dSrc;
    delete[] rgb;
}

int main(int argc, char* argv[]) {
    //testConvertToP208();
    testRGBConvert();
    return 0;
}