#include "string_kernels.cu"

void cudaVLikeCWarpWrapper(const char *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount,const size_t numBlocks, const size_t numThreads) {
    cudaVLikeCWarp <<<numBlocks,numThreads>>> (data,startIndex,endIndex,numElements,searchSlotsLen,matchCount);
}


