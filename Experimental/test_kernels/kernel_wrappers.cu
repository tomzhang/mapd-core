#include "string_kernels.cu"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

unsigned int populateSearchVals(const string& searchString, vector<vector<unsigned int>>& searchVals) {
  searchVals.resize(4);
  size_t stringSize = searchString.size();
  vector<unsigned int> offsetArray;  // = {2 << 24, 2<<16, 2<< 8, 1};
  offsetArray.push_back(1);          // 2^0
  offsetArray.push_back(256);        // 2^8
  offsetArray.push_back(65536);      // 2^16
  offsetArray.push_back(16777216);  // 2^24
  unsigned int maxLen = 0;
  for (int offset = 0; offset < 4; ++offset) {
    // int startSize = 4 - offset;
    int startPos = offset;
    size_t charPos = 0;
    unsigned int val = 0;
    while (startPos < 4 && charPos < stringSize) {
      unsigned int inc = static_cast<unsigned int>(searchString[charPos]) * offsetArray[startPos];
      // cout << "Inc " << charPos << ": " << inc << endl;
      val += inc;
      charPos++;
      startPos++;
    }
    searchVals[offset].push_back(val);
    val = 0;
    size_t slot = 0;
    while (charPos < stringSize) {
      val += static_cast<int>(searchString[charPos]) * offsetArray[slot];
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
  /*
  for (int offset = 0; offset < 4; ++offset) {
      if (searchVals[offset].size() < maxLen) { //should only be by 1
          searchVals[offset].push_back(0);
      }
  }
  */
  return maxLen;
}

void cudaVLikeCWarpWrapper(const char* data,
                           const unsigned int* startIndex,
                           const unsigned int* endIndex,
                           const unsigned int numElements,
                           unsigned int* matchCount,
                           const size_t numBlocks,
                           const size_t numThreads,
                           const string& searchString) {
  vector<vector<unsigned int>> searchVals;
  unsigned int searchSize = populateSearchVals(searchString, searchVals);
  for (int offset = 0; offset < 4; ++offset) {
    for (int v = 0; v != searchVals[offset].size(); ++v) {
      cout << searchVals[offset][v] << " ";
    }
    cout << endl;
  }
  size_t searchStringSize = searchString.size();

  // unsigned int hostSearchData = 1701336074;

  // unsigned int hostSearchData = 1701603616;
  // cudaMemcpyToSymbol(searchData, &hostSearchData, sizeof(unsigned int));
  unsigned int startBitmask = 0xFFFFFFFF;
  cudaMemcpyToSymbol(startBitmasks, &startBitmask, sizeof(unsigned int));
  startBitmask = 0xFFFFFF00;
  cudaMemcpyToSymbol(startBitmasks, &startBitmask, sizeof(unsigned int), 4);
  startBitmask = 0xFFFF0000;
  cudaMemcpyToSymbol(startBitmasks, &startBitmask, sizeof(unsigned int), 8);
  startBitmask = 0xFF000000;
  cudaMemcpyToSymbol(startBitmasks, &startBitmask, sizeof(unsigned int), 12);

  for (int offset = 0; offset < 4; ++offset) {
    unsigned int vecSize = searchVals[offset].size();
    cudaMemcpyToSymbol(numSlots, &vecSize, sizeof(unsigned int), offset * sizeof(unsigned int));
    unsigned int modVal = (searchStringSize + offset) % 4;
    unsigned int endBitmask;
    switch (modVal) {
      case 0:
        endBitmask = 0xFFFFFFFF;
        break;
      case 1:
        endBitmask = 0x000000FF;
        break;
      case 2:
        endBitmask = 0x0000FFFF;
        break;
      case 3:
        endBitmask = 0x00FFFFFF;
        break;
    }
    cudaMemcpyToSymbol(endBitmasks, &endBitmask, sizeof(unsigned int), offset * sizeof(unsigned int));
    cudaMemcpyToSymbol(
        searchData, &(searchVals[offset][0]), sizeof(unsigned int) * vecSize, offset * 8 * sizeof(unsigned int));
  }

  cudaMemset(matchCount, 0, sizeof(unsigned int));
  cudaVLikeCWarp2 << <numBlocks, numThreads>>>
      ((unsigned int*)data, startIndex, endIndex, numElements, searchString.size(), matchCount);
}
