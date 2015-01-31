#include "CpuBufferMgr.h"
#include "CpuBuffer.h"
#include "../../../CudaMgr/CudaMgr.h"

namespace Buffer_Namespace {

    CpuBufferMgr::CpuBufferMgr(const size_t maxBufferSize, const CpuBufferMgrMemType cpuBufferMgrMemType, CudaMgr_Namespace::CudaMgr *cudaMgr, const size_t bufferAllocIncrement,  const size_t pageSize, AbstractBufferMgr *parentMgr) : BufferMgr(maxBufferSize, bufferAllocIncrement, pageSize, parentMgr), cpuBufferMgrMemType_(cpuBufferMgrMemType), cudaMgr_(cudaMgr) {}

    CpuBufferMgr::~CpuBufferMgr() {
        freeAllMem();
    }

    void CpuBufferMgr::addSlab(const size_t slabSize) {
        slabs_.resize(slabs_.size()+1);
        if (cpuBufferMgrMemType_ == CUDA_HOST) {
            slabs_.back() = cudaMgr_->allocatePinnedHostMem(slabSize);
        }
        else {
            slabs_.back() = new int8_t[slabSize];
        }
        slabSegments_.resize(slabSegments_.size()+1);
        slabSegments_[slabSegments_.size()-1].push_back(BufferSeg(0,numPagesPerSlab_));
    }

    void CpuBufferMgr::freeAllMem() {
        for (auto bufIt = slabs_.begin(); bufIt != slabs_.end(); ++bufIt) { 
            if (cpuBufferMgrMemType_ == CUDA_HOST) {
                cudaMgr_->freePinnedHostMem(*bufIt);
            }
            else {
                delete [] *bufIt;
            }
        }
    }
    

    void CpuBufferMgr::allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t initialSize) {
        new CpuBuffer(this, segIt, cudaMgr_, pageSize, initialSize); // this line is admittedly a bit weird but the segment iterator passed into buffer takes the address of the new Buffer in its buffer member
    }


} // Buffer_Namespace
