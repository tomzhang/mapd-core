#include "file_utils.hpp"
#include "gpu_utils.hpp"
#include "kernel_wrappers.hpp"
#include <string>
#include <iostream>
#include <boost/timer/timer.hpp>

using namespace std;

int main (int argc, char **argv) {
    string colBaseName (argv[1]); 
    string startIndexFileName (colBaseName + ".indexstart"); 
    string endIndexFileName (colBaseName + ".indexend"); 
    string dataFileName (colBaseName + ".data"); 
    vector <unsigned int> startIndex;
    vector <unsigned int> endIndex;
    vector <char> data;
    setGpu(0);
    size_t multiProcessorCount = getMultiProcessorCount();
    cout << "Multiprocessor count: " << multiProcessorCount << endl;

    readBinaryFile(startIndexFileName,startIndex);
    readBinaryFile(endIndexFileName,endIndex);
    readBinaryFile(dataFileName,data);
    cout << "Start Index file length: " << startIndex.size() << endl;
    cout << "End Index file length: " << endIndex.size() << endl;
    cout << "Data file length: " << data.size() << endl;
    unsigned int *gpuStartIndexPtr = copyToGpu(startIndex);
    if (gpuStartIndexPtr == 0)
        cout << "Start Index copy failed!" << endl;
    unsigned int *gpuEndIndexPtr = copyToGpu(endIndex);
    if (gpuEndIndexPtr == 0)
        cout << "End Index copy failed!" << endl;
    char *gpuDataPtr = copyToGpu(data);
    if (gpuDataPtr == 0)
        cout << "Data copy failed!" << endl;
    unsigned int *devCountPtr;
    allocateGpuMem(devCountPtr,1);
    cout << "Data loaded" << endl;
    string searchString;
    do {
        cout << "Enter search string: ";
        getline(cin,searchString);
        if (searchString == "q")
            break;

        {
            const boost::timer::nanosecond_type oneSecond(1000000000LL);
            boost::timer::cpu_timer cpuTimer;
            cudaVLikeCWarpWrapper (gpuDataPtr,gpuStartIndexPtr,gpuEndIndexPtr,startIndex.size(),devCountPtr,multiProcessorCount*2,1024,searchString);
            unsigned int * hostCountPtr = copyFromGpu(devCountPtr,1);
            double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond * 1000.0;
            cout << "Elapsed time: " << elapsedTime << " ms." << endl;
            cout << "Count: " << *hostCountPtr << endl;
        }
    }
    while (1 == 1);
    cout << "Kernel Finished" << endl;


}




