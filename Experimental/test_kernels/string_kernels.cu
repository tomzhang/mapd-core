#define WARPSIZE 32

__constant__ unsigned int searchData[8];

__global__ void cudaVLikeCWarp(const unsigned int *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount, const unsigned int searchInt) { 
    __shared__ unsigned int sharedMatchCount;
    if (threadIdx.x == 0) {
        sharedMatchCount = 0;
    }
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int warpId = threadIdx.x / WARPSIZE;
    int warpThreadId = threadIdx.x % WARPSIZE;
    //int warpStride = 4096 * WARPSIZE / blockDim.x;
    //int warpOutOffset = warpId * warpStride;
    unsigned int localMatchCount = 0;
    while (i < numElements - 10000) {
        unsigned int indexStart = startIndex[i];
        unsigned int indexEnd = endIndex[i] - searchSlotsLen;
        int slotStart = indexStart / 4; 
        int slotEnd = indexEnd / 4; 
        int slotMod = indexEnd % 4;
        //int docSlotStart = indexStart / 4;
        //int docSlotEnd = indexEnd  / 4 + (indexEnd % 4 != 0);
        for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
            int docStart = __shfl(slotStart,curWarpDoc,WARPSIZE);
            int docEnd = __shfl(slotEnd,curWarpDoc,WARPSIZE);
            int docMod = __shfl(slotMod,curWarpDoc,WARPSIZE);
            bool found = false;
            unsigned int offset = docStart + warpThreadId;
            while (offset < docEnd) {
                if (data[offset] == searchInt)
                    found = true;
                offset += WARPSIZE;
            }

            int warpFound = __any(found);
            if (warpThreadId == 0 && warpFound) {
                localMatchCount++;
            }
            //int numWarpCycles = (docEnd-docStart+WARPSIZE)/WARPSIZE; // because end is inclusive here
            //numCycles =    
            //for (unsigned int cIndex = docStart + warpThreadId; cIndex < lastIndex; cIndex += WARPSIZE) {
        }

        i += stride;
    }
    if (warpThreadId == 0) 
        atomicAdd(matchCount, localMatchCount);
}




