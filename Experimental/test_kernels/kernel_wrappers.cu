#include "string_kernels.cu"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

void populateSearchVals(const string &searchString, vector<vector<unsigned int> > &searchVals) {
    searchVals.resize(4);
    size_t stringSize = searchString.size();
    vector <unsigned int> offsetArray;// = {2 << 24, 2<<16, 2<< 8, 1};
    offsetArray.push_back(1);
    offsetArray.push_back(256);
    offsetArray.push_back(65536);
    offsetArray.push_back(16777216);
    for (int offset = 0; offset < 4; ++offset) {
        int startSize = 4 - offset;
        size_t charPos = 0;
        unsigned int val = 0;
        while (startSize > 0 &&  charPos < stringSize) {
            unsigned int inc = static_cast<unsigned int> (searchString[charPos])*offsetArray[charPos]; 
            //cout << "Inc " << charPos << ": " << inc << endl;
            val += inc;
            charPos++;
            startSize--;
        }
        searchVals[offset].push_back(val);
        val = 0;
        size_t slot = 0;
        while (charPos < stringSize) {
            val += static_cast<int>(searchString[charPos])*offsetArray[slot];
            charPos++;
            slot++;
            if (slot == 4) {
                searchVals[offset].push_back(val);
                slot = 0;
            }
        }
        if (slot != 0) {
            searchVals[offset].push_back(val);
        }
    }
}

void cudaVLikeCWarpWrapper(const char *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, unsigned int * matchCount,const size_t numBlocks, const size_t numThreads, const string &searchString) {

    vector<vector<unsigned int> > searchVals;
    populateSearchVals(searchString, searchVals);
    for (int offset = 0; offset < 4; ++offset) {
        for (int v = 0; v != searchVals[offset].size(); ++v) {
            cout << searchVals[offset][v] << " ";
        }
        cout << endl;
    }

    //unsigned int hostSearchData = 1701336074;

    //unsigned int hostSearchData = 1701603616;
    //cudaMemcpyToSymbol(searchData, &hostSearchData, sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData, &searchVals[0][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData + 32, &searchVals[1][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData + 64, &searchVals[2][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData + 96, &searchVals[3][0], sizeof(unsigned int)); 
    /*
    cudaMemcpyToSymbol(searchData0, &searchVals[0][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData1, &searchVals[1][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData2, &searchVals[2][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData3, &searchVals[3][0], sizeof(unsigned int)); 
    */
    cudaMemset(matchCount,0,sizeof(unsigned int));
    cudaVLikeCWarp <<<numBlocks,numThreads>>> ((unsigned int *) data,startIndex,endIndex,numElements,0,matchCount);
}


