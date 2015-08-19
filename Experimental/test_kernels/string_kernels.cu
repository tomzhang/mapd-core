#define WARPSIZE 32
#include <stdio.h>
#include <cuda.h>

__constant__ unsigned int searchData[32];
__constant__ unsigned int startBitmasks[4];
__constant__ unsigned int endBitmasks[4];
__constant__ unsigned int numSlots[4];
//__constant__ unsigned int searchData0[8];
//__constant__ unsigned int searchData1[8];
//__constant__ unsigned int searchData2[8];
//__constant__ unsigned int searchData3[8];

#define SEARCH_OFFSET 32

//__global__ void cudaVLikeCWarp(const unsigned int *data, const unsigned int *startIndex, const unsigned int *endIndex,
//const unsigned int numElements, const unsigned int searchSlotsLen, unsigned int * matchCount) {
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

//__global__ void cudaVLikeCWarp(const unsigned int *data, const unsigned int *startIndex, const unsigned int *endIndex,
//const unsigned int numElements, const unsigned int searchLen, unsigned int * matchCount) {
//    //__shared__ unsigned int sharedData [2048]; // 1024 threads / 32 threads/warp = 32 warps * 64 unsigned ints each
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
//    int warpNeighborId = (warpThreadId - 1) % 32;
//    //int warpStride = 2048 * WARPSIZE / blockDim.x; /64
//    int warpStride = 64;
//    int warpOutOffset = warpId * warpStride;
//    unsigned int searchSlotsLen = (searchLen + 2) / 4 + 1;
//    unsigned int localMatchCount = 0;
//    unsigned int sharedOffset = warpOutOffset + warpThreadId;
//    while (i - warpThreadId < numElements) {
//        unsigned int indexStart;
//        unsigned int indexEnd;
//        int slotStart;
//        int slotEnd;
//        //int slotMod;
//        if (i < numElements) {
//            indexStart = startIndex[i];
//            //indexEnd = endIndex[i] - searchSlotsLen;
//            indexEnd = endIndex[i];
//            slotStart = indexStart / 4;
//            slotEnd = indexEnd / 4;
//            //slotMod = indexEnd % 4;
//        }
//        else { // ensure that this non-existent doc won't be read
//            slotStart = 1;
//            slotEnd = 0;
//        }
//        for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
//            int docStart = __shfl(slotStart,curWarpDoc,WARPSIZE);
//            int docEnd = __shfl(slotEnd,curWarpDoc,WARPSIZE);
//            //int docMod = __shfl(slotMod,curWarpDoc,WARPSIZE);
//            unsigned int offset = docStart + warpThreadId;
//            unsigned int localData = 0;
//            if (offset < docEnd) {
//                localData = data[offset];
//            }
//            unsigned int nextData = 0;
//            int found;
//            //int warpFound = 0;
//            while (offset < docEnd) {
//                int iteration = 0;
//                if (offset + WARPSIZE < docEnd) {
//                    nextData = data[offset+WARPSIZE];
//                }
//                else {
//                    nextData = 0;
//                }
//                found = 15;
//                int warpFound = 1;
//                while (warpFound && iteration < searchSlotsLen) {
//                    if (warpThreadId < iteration) {
//                        localData = nextData;
//                    }
//                    int newFound = 0;
//                    if (localData > 0) {
//                        if (iteration == 0) {
//                            for (int charOffset = 0; charOffset < 4; ++charOffset) {
//                                unsigned int searchDataLocal = searchData[charOffset*8+iteration];
//                                localData = localData & bitmasks[charOffset];
//                                newFound += ((1 << charOffset) * (localData == searchDataLocal));
//                            }
//                        }
//                        /*
//                        else {
//                            for (int charOffset = 0; charOffset < 4; ++charOffset) {
//                                unsigned int searchDataLocal = searchData[charOffset*8+iteration];
//                                newFound += ((1 << charOffset) * (localData == searchDataLocal));
//                            }
//                        }
//                        */
//
//                        //for (int charOffset = 0; charOffset < 4; ++charOffset) {
//                        //    if ((found & (1 << charOffset))) {
//                        //        unsigned int searchDataLocal = searchData[charOffset*8+iteration];
//                        //        /*
//                        //        if (warpId == 0 && i < 32) {
//                        //            printf ("Doc: %d Thread: %d Offset: %d CharOffset: %d Iteration: %d SearchData:
//                        %u\n",curWarpDoc,i,offset,charOffset,iteration,searchDataLocal);
//                        //        }
//                        //        */
//                        //        if (iteration == 0) {
//                        //            localData = localData & bitmasks[charOffset];
//                        //        }
//                        //        /*
//                        //        if (warpId == 0 && i < 32) {
//                        //            printf("Anding local: %u => %u\n",localData,andVal);
//                        //        }
//                        //        */
//                        //        if (localData == searchDataLocal || searchDataLocal == 0) {
//                        //             newFound += (1 << charOffset);
//                        //            /*
//                        //            if (warpId == 0 && i < 32) {
//                        //                printf("Newfound Thread: %d Offset: %d: Newfound:
//                        %d\n",i,charOffset,newFound);
//                        //            }
//                        //            */
//                        //         }
//                        //    }
//                        //}
//                    }
//                    found = newFound;
//                    /*
//                    if (warpId == 0 && i < 32 && found > 0)
//                        printf("Doc: %d Thread: %d WarpNeighborId: %d Iteration: %d Found: %d\n", curWarpDoc, i,
//                        warpNeighborId , iteration,found);
//                    */
//                    iteration++;
//                    warpFound = __any(found);
//                    found = __shfl(found,warpNeighborId,WARPSIZE);
//                }
//                if (warpFound) {
//                    if (warpThreadId == 0) {
//                    /*
//                    if (i == 0) {
//                        printf("Doc: %d => String FOUND!\n",curWarpDoc);
//                    }
//                    */
//
//                    localMatchCount++;
//
//                    }
//                    break;
//                }
//                offset += WARPSIZE;
//                localData = nextData;
//            }
//        }
//
//        i += stride;
//    }
//    if (warpThreadId == 0)
//        atomicAdd(matchCount, localMatchCount);
//}

__global__ void cudaVLikeCWarp2(const unsigned int* data,
                                const unsigned int* startIndex,
                                const unsigned int* endIndex,
                                const unsigned int numElements,
                                const unsigned int searchLen,
                                unsigned int* matchCount) {
  __shared__ unsigned int sharedData[2048];  // 1024 threads / 32 threads/warp = 32 warps * 64 unsigned ints each
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
  /*
  int warpNeighborId = (warpThreadId - 1) % 32;
  */
  // int warpStride = 2048 * WARPSIZE / blockDim.x; /64
  int warpStride = 64;
  int warpOutOffset = warpId * warpStride;
  // unsigned int searchSlotsLen = (searchLen + 2) / 4 + 1;
  unsigned int localMatchCount = 0;
  unsigned int sharedOffset = warpOutOffset + warpThreadId;
  while (i - warpThreadId < numElements) {
    int slotStart;
    int slotEnd;
    // int slotMod;
    if (i < numElements) {
      slotStart = startIndex[i] / 4;
      slotEnd = endIndex[i] / 4;
    } else {  // ensure that this non-existent doc won't be read
      slotStart = 1;
      slotEnd = 0;
    }
    for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
      int docStart = __shfl(slotStart, curWarpDoc, WARPSIZE);
      int docEnd = __shfl(slotEnd, curWarpDoc, WARPSIZE);
      unsigned int offset = docStart + warpThreadId;
      while (offset < docEnd) {
        unsigned int localData = data[offset];
        unsigned int found = 0;
        unsigned int foundOffset = 1;
        for (int charOffset = 0; charOffset < 4; ++charOffset) {
          unsigned int numSearchSlots = numSlots[charOffset] - 1;
          unsigned int iterationData = localData;
          iterationData = iterationData & startBitmasks[charOffset];
          if (numSearchSlots == 0)
            iterationData = iterationData & endBitmasks[charOffset];
          found += foundOffset * (iterationData == searchData[charOffset * 8]);
          for (int iteration = 1; iteration <= numSearchSlots; ++iteration) {
            iterationData = localData;
            if (iteration == numSearchSlots)
              iterationData = iterationData & endBitmasks[charOffset];
            found += foundOffset * (iterationData == searchData[charOffset * 8 + iteration]);
            foundOffset *= 2;
            localMatchCount++;
          }
        }
        sharedData[sharedOffset] = found;
        offset += WARPSIZE;
      }
    }
    i += stride;
  }
  atomicAdd(matchCount, localMatchCount);
}

__global__ void cudaVLikeCWarp2(const unsigned int* data,
                                const unsigned int* startIndex,
                                const unsigned int* endIndex,
                                const unsigned int numElements,
                                const unsigned int searchLen,
                                unsigned int* matchCount) {
  __shared__ unsigned int sharedData[2048];  // 1024 threads / 32 threads/warp = 32 warps * 64 unsigned ints each
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
  /*
  int warpNeighborId = (warpThreadId - 1) % 32;
  */
  // int warpStride = 2048 * WARPSIZE / blockDim.x; /64
  int warpStride = 64;
  int warpOutOffset = warpId * warpStride;
  // unsigned int searchSlotsLen = (searchLen + 2) / 4 + 1;
  unsigned int localMatchCount = 0;
  unsigned int sharedOffset = warpOutOffset + warpThreadId;
  while (i - warpThreadId < numElements) {
    int slotStart;
    int slotEnd;
    // int slotMod;
    if (i < numElements) {
      slotStart = startIndex[i] / 4;
      slotEnd = endIndex[i] / 4;
    } else {  // ensure that this non-existent doc won't be read
      slotStart = 1;
      slotEnd = 0;
    }
    for (int curWarpDoc = 0; curWarpDoc != 32; ++curWarpDoc) {
      int docStart = __shfl(slotStart, curWarpDoc, WARPSIZE);
      int docEnd = __shfl(slotEnd, curWarpDoc, WARPSIZE);
      unsigned int offset = docStart + warpThreadId;
      while (offset < docEnd) {
        unsigned int localData = data[offset];
        unsigned int found = 0;
        unsigned int foundOffset = 1;
        for (int charOffset = 0; charOffset < 4; ++charOffset) {
          unsigned int numSearchSlots = numSlots[charOffset] - 1;
          unsigned int iterationData = localData;
          iterationData = iterationData & startBitmasks[charOffset];
          if (numSearchSlots == 0)
            iterationData = iterationData & endBitmasks[charOffset];
          found += foundOffset * (iterationData == searchData[charOffset * 8]);
          for (int iteration = 1; iteration <= numSearchSlots; ++iteration) {
            iterationData = localData;
            if (iteration == numSearchSlots)
              iterationData = iterationData & endBitmasks[charOffset];
            found += foundOffset * (iterationData == searchData[charOffset * 8 + iteration]);
            foundOffset *= 2;
            localMatchCount++;
          }
        }
        sharedData[sharedOffset] = found;
        offset += WARPSIZE;
      }
    }
    i += stride;
  }
  atomicAdd(matchCount, localMatchCount);
}
