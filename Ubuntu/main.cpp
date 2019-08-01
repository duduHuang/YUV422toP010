#include <iostream>

int convert(int argc, char** argv);
bool isGPUEnable();

using namespace std;

int main(int argc, char* argv[]) {

    if (isGPUEnable()) {
		convert(argc, argv);
	}
    return 0;
}