#define WARPSIZE 32
#include <stdio.h>
#include <cuda.h>

__constant__ unsigned int searchData[128];
//__constant__ unsigned int searchData0[8];
//__constant__ unsigned int searchData1[8];
//__constant__ unsigned int searchData2[8];
//__constant__ unsigned int searchData3[8];

__global__ void cudaVLikeCWarp(const unsigned int *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount) { 
    __shared__ unsigned int sharedData [2048]; // 1024 threads / 32 threads/warp = 32 warps * 64 unsigned ints each
    /*
    if (threadIdx.x == 0) {
        sharedMatchCount = 0;
    }
    __syncthreads();
    */
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int warpId = threadIdx.x / WARPSIZE;
    int warpThreadId = threadIdx.x % WARPSIZE;
    //int warpStride = 2048 * WARPSIZE / blockDim.x; /64
    int warpStride = 64;
    int warpOutOffset = warpId * warpStride;
    unsigned int localMatchCount = 0;
    unsigned int sharedOffset = warpOutOffset + warpThreadId;
    while (i - warpThreadId < numElements) {
    //while (i < 248736) {
        unsigned int indexStart;
        unsigned int indexEnd;
        int slotStart;
        int slotEnd;
        int slotMod;
        if (i < numElements) {
            indexStart = startIndex[i];
            indexEnd = endIndex[i] - searchSlotsLen;
            slotStart = indexStart / 4; 
            slotEnd = indexEnd / 4; 
            slotMod = indexEnd % 4;
        }
        else { // ensure that this non-existent doc won't be read
            slotStart = 1;
            slotEnd = 0;
        }
            /*
        unsigned int indexStart = startIndex[i];
        unsigned int indexEnd = endIndex[i] - searchSlotsLen;
        int slotStart = indexStart / 4; 
        int slotEnd = indexEnd / 4; 
        int slotMod = indexEnd % 4;
        */
        /*
        if (i < 10)
            printf("%u-%u: %u\n", indexStart, indexEnd,data[slotStart]);
        */
        //int docSlotStart = indexStart / 4;
        //int docSlotEnd = indexEnd  / 4 + (indexEnd % 4 != 0);
        for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
            int docStart = __shfl(slotStart,curWarpDoc,WARPSIZE);
            int docEnd = __shfl(slotEnd,curWarpDoc,WARPSIZE);
            int docMod = __shfl(slotMod,curWarpDoc,WARPSIZE);
            bool found = false;
            unsigned int offset = docStart + warpThreadId;
            while (offset <= docEnd) {
                sharedData[sharedOffset] = data[offset];
                unsigned int localSharedData = sharedData[sharedOffset];
                if (localSharedData == searchData[0])
                    found = true;
                localSharedData = localSharedData >> 8; 
                if (localSharedData == searchData[32])
                    found = true;
                localSharedData = localSharedData >> 8; 
                if (localSharedData == searchData[64])
                    found = true;
                localSharedData = localSharedData >> 8; 
                if (localSharedData == searchData[96])
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
    /*
    if (warpThreadId == 0)
        atomicAdd( &(sharedMatchCount), localMatchCount);
    __syncthreads();
    */
    if (warpThreadId == 0) 
        atomicAdd(matchCount, localMatchCount);
}




