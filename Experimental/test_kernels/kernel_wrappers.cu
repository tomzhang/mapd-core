#include "string_kernels.cu"

void cudaVLikeCWarpWrapper(const char *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount,const size_t numBlocks, const size_t numThreads) {

   unsigned int hostSearchData = 1701336074;
   //unsigned int hostSearchData = 1701603616;
   cudaMemcpyToSymbol(searchData, &hostSearchData, sizeof(unsigned int)); 

    cudaVLikeCWarp <<<numBlocks,numThreads>>> ((unsigned int *) data,startIndex,endIndex,numElements,searchSlotsLen,matchCount, hostSearchData);
}


