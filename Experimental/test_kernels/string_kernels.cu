#define WARPSIZE 32

__global__ void cudaVLikeCWarp(const char *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount) { 
    __shared__ unsigned int sharedMatchCount;
    if (threadIdx.x == 0) {
        sharedMatchCount = 0;
    }
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int warpId = threadIdx.x / WARPSIZE;
    int warpThreadId = threadIdx.x % WARPSIZE;
    int warpStride = 4096 * WARPSIZE / blockDim.x;
    int warpOutOffset = warpId * warpStride;
    unsigned int localMatchCount = 0;
    while (i < numElements) {
        unsigned int indexStart = startIndex[i];
        unsigned int indexEnd = endIndex[i] - searchSlotsLen;
        unsigned int docSlotStart = indexStart / 4;
        unsigned int docSlotEnd = indexEnd  / 4 + (indexEnd % 4 != 0);
        localMatchCount++;
        for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
            /*
            unsigned int docStart = __shfl(int(readStart),curWarpDoc,WARPSIZE);
            unsigned int docEnd = __shfl(int(readEnd),curWarpDoc,WARPSIZE);
            unsigned int lastIndex = docEnd - searchSlotsLen; 
            */
            bool found = false;

        }
        i += stride;
    }
    atomicAdd(matchCount, localMatchCount);
}




