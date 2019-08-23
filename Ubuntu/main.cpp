#include <iostream>
#include "convertTool.h"

using namespace std;

int main(int argc, char* argv[]) {
    unsigned char *r = new unsigned char[1280 * 720 * 2];
    ConverterTool *converterTool;
    converterTool = new ConverterTool();
    if (converterTool->isGPUEnable()) {
        int i = 1;
        converterTool->initialCuda();
        converterTool->lookupTableF();
        converterTool->setSrcSize(7680, 4320);
        converterTool->setDstSize(1280, 720);
        converterTool->preprocess();
        converterTool->allocateMem();
        while (i) {
            converterTool->testFunction();
            cout << "continue ? ";
            cin >> i;
        }
    }
    else {
        cout << "device hasn't cuda !!!\n";
    }
    converterTool->freeMemory();
    converterTool->destroyCudaEvent();
    delete[] r;
    return 0;
}