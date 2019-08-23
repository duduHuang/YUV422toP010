#include <iostream>
#include "convertTool.h"

using namespace std;

int main(int argc, char* argv[]) {
    unsigned short *dSrc;
    int frameSize = (7680 + 47) / 48 * 128 * 4320;
    dSrc = new unsigned short[frameSize];
    ifstream v210File("v210.yuv", ifstream::in | ios::binary);
    if (!v210File.is_open()) {
        cerr << "Can't open files\n";
        return -1;
    }
    v210File.read((char *)dSrc, frameSize);
    if (v210File.gcount() < frameSize) {
        cerr << "can't get one frame\n";
        return -1;
    }
    v210File.close();

    unsigned char *p208;
    p208 = new unsigned char[1280 * 720 * 2];
    int nJPEGSize = 0;

    ConverterTool *converterTool;
    converterTool = new ConverterTool();
    if (converterTool->isGPUEnable()) {
        int i = 1;
        converterTool->initialCuda();
        converterTool->lookupTableF();
        converterTool->setSrcSize(7680, 4320);
        converterTool->setDstSize(1280, 720);
        converterTool->allocateMem();
        while (i) {
            converterTool->convertToP208ThenResize(dSrc, p208, &nJPEGSize);
            cout << "continue ? ";
            cin >> i;
        }
    }
    else {
        cout << "device hasn't cuda !!!\n";
    }
    ofstream output_file("r.jpg", ios::out | ios::binary);
    output_file.write((char *)p208, nJPEGSize);
    output_file.close();
    delete[] p208;
    converterTool->freeMemory();
    converterTool->destroyCudaEvent();
    delete[] dSrc;
    return 0;
}