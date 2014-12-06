#include "string_kernels.cu"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

unsigned int populateSearchVals(const string &searchString, vector<vector<unsigned int> > &searchVals) {
    searchVals.resize(4);
    size_t stringSize = searchString.size();
    vector <unsigned int> offsetArray;// = {2 << 24, 2<<16, 2<< 8, 1};
    offsetArray.push_back(1); // 2^0
    offsetArray.push_back(256); // 2^8
    offsetArray.push_back(65536); // 2^16
    offsetArray.push_back(16777216); //2^24
    unsigned int maxLen = 0;
    for (int offset = 0; offset < 4; ++offset) {
        //int startSize = 4 - offset;
        int startPos = offset;
        size_t charPos = 0;
        unsigned int val = 0;
        while (startPos < 4 &&  charPos < stringSize) {
            unsigned int inc = static_cast<unsigned int> (searchString[charPos])*offsetArray[startPos]; 
            //cout << "Inc " << charPos << ": " << inc << endl;
            val += inc;
            charPos++;
            startPos++;
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
                val = 0;
                slot = 0;
            }
        }
        if (slot != 0) {
            searchVals[offset].push_back(val);
        }
        if (searchVals[offset].size() > maxLen) {
            maxLen = searchVals[offset].size();
        }
    }
    for (int offset = 0; offset < 4; ++offset) {
        if (searchVals[offset].size() < maxLen) { //should only be by 1
            searchVals[offset].push_back(0);
        }
    }
    return maxLen;
}

void cudaVLikeCWarpWrapper(const char *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, unsigned int * matchCount,const size_t numBlocks, const size_t numThreads, const string &searchString) {

    vector<vector<unsigned int> > searchVals;
    unsigned int searchSize = populateSearchVals(searchString, searchVals);
    for (int offset = 0; offset < 4; ++offset) {
        for (int v = 0; v != searchVals[offset].size(); ++v) {
            cout << searchVals[offset][v] << " ";
        }
        cout << endl;
    }


    //unsigned int hostSearchData = 1701336074;

    //unsigned int hostSearchData = 1701603616;
    //cudaMemcpyToSymbol(searchData, &hostSearchData, sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData, &(searchVals[0][0]), sizeof(unsigned int) * searchSize); 
    cudaMemcpyToSymbol(searchData, &(searchVals[1][0]), sizeof(unsigned int) * searchSize,32); 
    cudaMemcpyToSymbol(searchData, &(searchVals[2][0]), sizeof(unsigned int) * searchSize,64); 
    cudaMemcpyToSymbol(searchData, &(searchVals[3][0]), sizeof(unsigned int) * searchSize,96); 
    /*
    cudaMemcpyToSymbol(searchData0, &searchVals[0][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData1, &searchVals[1][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData2, &searchVals[2][0], sizeof(unsigned int)); 
    cudaMemcpyToSymbol(searchData3, &searchVals[3][0], sizeof(unsigned int)); 
    */
    cudaMemset(matchCount,0,sizeof(unsigned int));
    cudaVLikeCWarp <<<numBlocks,numThreads>>> ((unsigned int *) data,startIndex,endIndex,numElements,searchSize,matchCount);
}


