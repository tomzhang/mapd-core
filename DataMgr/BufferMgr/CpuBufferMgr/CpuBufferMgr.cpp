#include "CpuBufferMgr.h"
#include "CpuBuffer.h"
#include "../CudaUtils.h"

namespace Buffer_Namespace {

    CpuBufferMgr::CpuBufferMgr(const size_t maxBufferSize, const CpuBufferMgrMemType cpuBufferMgrMemType, const size_t bufferAllocIncrement,  const size_t pageSize, AbstractDataMgr *parentMgr) : BufferMgr(maxBufferSize, bufferAllocIncrement, pageSize, parentMgr), cpuBufferMgrMemType_(cpuBufferMgrMemType) {}

    CpuBufferMgr::~CpuBufferMgr() {
        freeAllMem();
    }

    void CpuBufferMgr::addSlab(const size_t slabSize) {
        slabs_.resize(slabs_.size()+1);
        if (cpuBufferMgrMemType_ == CUDA_HOST) {
            CudaUtils::allocPinnedHostMem(slabs_.back(),slabSize,1);
        }
        else {
            slabs_.back() = (mapd_addr_t) new mapd_byte_t[slabSize];
        }
        //slabs_.push_back((mapd_addr_t) malloc(slabSize));
        slabSegments_.resize(slabSegments_.size()+1);
        slabSegments_[slabSegments_.size()-1].push_back(BufferSeg(0,numPagesPerSlab_));
    }

    void CpuBufferMgr::freeAllMem() {
        for (auto bufIt = slabs_.begin(); bufIt != slabs_.end(); ++bufIt) { 
            if (cpuBufferMgrMemType_ == CUDA_HOST) {
                CudaUtils::hostFree(*bufIt);
            }
            else {
                delete [] *bufIt;
            }
        }
    }
    /*
    void BufferMgr::fetchChunk(const ChunkKey &key, AbstractBuffer *destBuffer, const mapd_size_t numBytes) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt == chunkIndex_.end()) 
            throw std::runtime_error("Chunk does not exist");
     */       




    void CpuBufferMgr::createBuffer(BufferList::iterator segIt, const mapd_size_t pageSize, const mapd_size_t initialSize) {
        new CpuBuffer(this, segIt, pageSize, initialSize); // this line is admittedly a bit weird but the segment iterator passed into buffer takes the address of the new Buffer in its buffer member
    }


} // Buffer_Namespace
