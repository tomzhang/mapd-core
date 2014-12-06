#define WARPSIZE 32
#include <stdio.h>
#include <cuda.h>

__constant__ unsigned int searchData[128];
//__constant__ unsigned int searchData0[8];
//__constant__ unsigned int searchData1[8];
//__constant__ unsigned int searchData2[8];
//__constant__ unsigned int searchData3[8];

#define SEARCH_OFFSET 32

//__global__ void cudaVLikeCWarp(const unsigned int *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount) { 
//    __shared__ unsigned int sharedData [2048]; // 1024 threads / 32 threads/warp = 32 warps * 64 unsigned ints each
//    /*
//    if (threadIdx.x == 0) {
//        sharedMatchCount = 0;
//    }
//    __syncthreads();
//    */
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x;
//    int warpId = threadIdx.x / WARPSIZE;
//    int warpThreadId = threadIdx.x % WARPSIZE;
//    //int warpStride = 2048 * WARPSIZE / blockDim.x; /64
//    int warpStride = 64;
//    int warpOutOffset = warpId * warpStride;
//    unsigned int localMatchCount = 0;
//    unsigned int sharedOffset = warpOutOffset + warpThreadId;
//    while (i - warpThreadId < numElements) {
//    //while (i < 248736) {
//        unsigned int indexStart;
//        unsigned int indexEnd;
//        int slotStart;
//        int slotEnd;
//        int slotMod;
//        if (i < numElements) {
//            indexStart = startIndex[i];
//            indexEnd = endIndex[i] - searchSlotsLen;
//            slotStart = indexStart / 4; 
//            slotEnd = indexEnd / 4; 
//            slotMod = indexEnd % 4;
//        }
//        else { // ensure that this non-existent doc won't be read
//            slotStart = 1;
//            slotEnd = 0;
//        }
//            /*
//        unsigned int indexStart = startIndex[i];
//        unsigned int indexEnd = endIndex[i] - searchSlotsLen;
//        int slotStart = indexStart / 4; 
//        int slotEnd = indexEnd / 4; 
//        int slotMod = indexEnd % 4;
//        */
//        /*
//        if (i < 10)
//            printf("%u-%u: %u\n", indexStart, indexEnd,data[slotStart]);
//        */
//        //int docSlotStart = indexStart / 4;
//        //int docSlotEnd = indexEnd  / 4 + (indexEnd % 4 != 0);
//        for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
//            int docStart = __shfl(slotStart,curWarpDoc,WARPSIZE);
//            int docEnd = __shfl(slotEnd,curWarpDoc,WARPSIZE);
//            int docMod = __shfl(slotMod,curWarpDoc,WARPSIZE);
//            unsigned int offset = docStart + warpThreadId;
//            if (offset < docEnd) {
//                sharedData[sharedOffset] = data[offset];
//            }
//            while (offset <= docEnd) {
//                if (offset + WARPSIZE < docEnd) 
//                    sharedData[sharedOffset+WARPSIZE] = data[offset+WARPSIZE];
//                }
//                unsigned int found = 15; // 2^24+2^16+2^8+1
//                bool atFirst = true;
//                do {
//                    unsigned int localSharedData = sharedData[sharedOffset];
//                    for (int offset = 0; offset < 4; ++offset) {
//                        unsigned int tempData = localSharedData;
//                        if (atFirst && offset > 0) {
//                            sharedData &= rightMasks[offset];
//                        }
//                        
//                        if (localSharedData == searchData[offset*8])
//                            found &= 1 << offset;  
//                        }
//                    }
//                    afFirst = false;
//                }
//                while (found > 0);
//
//                // we are at startOffset 
//                /*
//                if (localSharedData == searchData[0])
//                    found = true;
//                localSharedData = localSharedData >> 8; 
//                if (localSharedData == searchData[32])
//                    found = true;
//                localSharedData = localSharedData >> 8; 
//                if (localSharedData == searchData[64])
//                    found = true;
//                localSharedData = localSharedData >> 8; 
//                if (localSharedData == searchData[96])
//                    found = true;
//                */
//
//
//
//                offset += WARPSIZE;
//            }
//
//            int warpFound = __any(found);
//            if (warpThreadId == 0 && warpFound) {
//                localMatchCount++;
//            }
//            //int numWarpCycles = (docEnd-docStart+WARPSIZE)/WARPSIZE; // because end is inclusive here
//            //numCycles =    
//            //for (unsigned int cIndex = docStart + warpThreadId; cIndex < lastIndex; cIndex += WARPSIZE) {
//        }
//
//        i += stride;
//    }
//    if (warpThreadId == 0) 
//        atomicAdd(matchCount, localMatchCount);
//}

__global__ void cudaVLikeCWarp(const unsigned int *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount) { 
    //__shared__ unsigned int sharedData [2048]; // 1024 threads / 32 threads/warp = 32 warps * 64 unsigned ints each
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
    int warpNeighborId = (warpThreadId - 1) % 32;
    //int warpStride = 2048 * WARPSIZE / blockDim.x; /64
    int warpStride = 64;
    int warpOutOffset = warpId * warpStride;
    unsigned int localMatchCount = 0;
    unsigned int sharedOffset = warpOutOffset + warpThreadId;
    while (i - warpThreadId < numElements) {
        unsigned int indexStart;
        unsigned int indexEnd;
        int slotStart;
        int slotEnd;
        //int slotMod;
        if (i < numElements) {
            indexStart = startIndex[i];
            //indexEnd = endIndex[i] - searchSlotsLen;
            indexEnd = endIndex[i];
            slotStart = indexStart / 4; 
            slotEnd = indexEnd / 4; 
            //slotMod = indexEnd % 4;
        }
        else { // ensure that this non-existent doc won't be read
            slotStart = 1;
            slotEnd = 0;
        }
        for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
            int docStart = __shfl(slotStart,curWarpDoc,WARPSIZE);
            int docEnd = __shfl(slotEnd,curWarpDoc,WARPSIZE);
            //int docMod = __shfl(slotMod,curWarpDoc,WARPSIZE);
            unsigned int offset = docStart + warpThreadId;
            unsigned int localData = 0;
            if (offset < docEnd) {
                localData = data[offset];
            }
            unsigned int nextData = 0;
            int found;
            while (offset < docEnd) {
                found = 15;
                int iteration = 0;
                if (offset + WARPSIZE < docEnd) {
                    nextData = data[offset+WARPSIZE];
                }
                else {
                    nextData = 0;
                }
                while (iteration < searchSlotsLen) {
                    if (warpThreadId < iteration) {
                        localData = nextData;
                    }
                    int newFound = 0;
                    if (localData > 0) {
                        for (int charOffset = 0; charOffset < 4; ++charOffset) {
                            if ((found & (1 << charOffset))) {
                                unsigned int searchDataLocal = searchData[charOffset*8+iteration];
                                /*
                                if (warpId == 0 && i < 32) {
                                    printf ("Doc: %d Thread: %d CharOffset: %d Iteration: %d SearchData: %u\n",curWarpDoc,i,charOffset,iteration,searchDataLocal); 
                                }
                                */
                                unsigned int andVal = localData & searchDataLocal;
                                /*
                                if (warpId == 0 && i < 32) {
                                    printf("Anding local: %u => %u\n",localData,andVal);
                                }
                                */
                                if (andVal == searchDataLocal) {
                                     newFound += (1 << charOffset);
                                    /*
                                    if (warpId == 0 && i < 32) {
                                        printf("Newfound %d: %d\n",charOffset,newFound);
                                    }
                                    */
                                 }
                            }
                        }
                    }
                    found = newFound;
                    /*
                    if (warpId == 0 && i < 32 && found > 0)
                        printf("Doc: %d Thread: %d WarpNeighborId: %d Iteration: %d Found: %d\n", curWarpDoc, i, warpNeighborId , iteration,found);
                    */
                    iteration++;
                    int warpMoreWork = __any(found);
                    if (!warpMoreWork)
                        break;
                    __shfl(found,warpNeighborId,WARPSIZE);
                }
                int warpFound = __any(found);
                if (warpThreadId == 0 && warpFound) {
                    localMatchCount++;
                }
                if (warpFound)
                    break;
                offset += WARPSIZE;
                localData = nextData;
            }
        }

        i += stride;
    }
    if (warpThreadId == 0) 
        atomicAdd(matchCount, localMatchCount);
}



