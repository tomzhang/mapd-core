//
//  Buffer.cpp
//  mapd2
//
//  @author Steven Stewart <steve@map-d.com>
//  @author Todd Mostak <todd@map-d.com>
//
//  Copyright (c) 2014 MapD Technologies, Inc. All rights reserved.
//
#include <cassert>
#include <stdexcept>

#include "Buffer.h"
#include "BufferMgr.h"


namespace Buffer_Namespace {

    Buffer::Buffer(BufferMgr *bm, BufferList::iterator segIt, const int deviceId, const size_t pageSize, const size_t numBytes): AbstractBuffer(deviceId), mem_(0), bm_(bm), segIt_(segIt), pageSize_(pageSize), numPages_(0), pinCount_(0) {
        // so that the pointer value of this Buffer is stored
        segIt_->buffer = this;
        if (numBytes > 0) {
            reserve(numBytes);
        }
    }


   /* 
    Buffer::Buffer(const int8_t * mem, const size_t numPages, const size_t pageSize, const int epoch):
     mem_(mem), pageSize_(pageSize), used_(0), epoch_(epoch), dirty_(false)
    {
        assert(pageSize_ > 0);
        pageDirtyFlags_.resize(numPages);
        for (size_t i = 0; i < numPages; ++i)
            pages_.push_back(Page(mem + (i * pageSize), false));
    
    }
    */
    
    Buffer::~Buffer() {
        
    }





    void Buffer::reserve(const size_t numBytes) {
#ifdef BUFFER_MUTEX
        boost::unique_lock < boost::shared_mutex > writeLock (readWriteMutex_);
#endif
        size_t numPages = (numBytes + pageSize_ -1 ) / pageSize_;
        //std::cout << "NumPages reserved: " << numPages << std::endl;
        if (numPages > numPages_) {
            pageDirtyFlags_.resize(numPages);
            numPages_ = numPages;
            segIt_ = bm_->reserveBuffer(segIt_,reservedSize());
        }
    }

    
    void Buffer::read(int8_t * const dst, const size_t numBytes, const size_t offset, const MemoryLevel dstBufferType, const int dstDeviceId) {
        assert(dst && mem_);
#ifdef BUFFER_MUTEX
        boost::shared_lock < boost::shared_mutex > readLock (readWriteMutex_);
#endif

        //std::cout << "Buffer size: " << size_ << std::endl;
        //std::cout << "Bytes to read: " << numBytes << std::endl;
        if (numBytes + offset > size_) {
            throw std::runtime_error("Buffer: Out of bounds read error");
        }
        readData(dst,numBytes,offset, dstBufferType,dstDeviceId);
        //memcpy(dst, mem_ + offset, numBytes);
    }

    void Buffer::write(int8_t * src, const size_t numBytes, const size_t offset, const MemoryLevel srcBufferType, const int srcDeviceId) {
        assert(numBytes > 0); // cannot write 0 bytes
#ifdef BUFFER_MUTEX
        boost::unique_lock < boost::shared_mutex > writeLock (readWriteMutex_);
#endif
        if (numBytes + offset > reservedSize()) {
            reserve(numBytes+offset);
            //bm_->reserveBuffer(segIt_,numBytes + offset);
        }
        //std::cout << "Size at beginning of write: " << size_ << std::endl;
        // write source contents to buffer
        //assert(mem_ && src);
        writeData(src,numBytes,offset,srcBufferType,srcDeviceId);
        //memcpy(mem_ + offset, src, numBytes);
        
        // update dirty flags for buffer and each affected page
        isDirty_ = true;
        if (offset < size_) {
            isUpdated_ = true;
        }
        if (offset + numBytes > size_) {
            isAppended_ = true;
            size_ = offset+numBytes;
        }
        //std::cout << "Size after write: " << size_ << std::endl;

        size_t firstDirtyPage = offset / pageSize_;
        size_t lastDirtyPage = (offset + numBytes - 1) / pageSize_;
        for (size_t i = firstDirtyPage; i <= lastDirtyPage; ++i) {
            pageDirtyFlags_[i] = true;
        }
    }

    void Buffer::append(int8_t * src, const size_t numBytes, const MemoryLevel srcBufferType, const int srcDeviceId) {
#ifdef BUFFER_MUTEX
        boost::shared_lock < boost::shared_mutex > readLock (readWriteMutex_); // keep another thread from getting a write lock
        boost::unique_lock < boost::shared_mutex > appendLock (appendMutex_); // keep another thread from getting an append lock
#endif


        isDirty_ = true;
        isAppended_ = true;


        if (numBytes + size_ > reservedSize()) {
            reserve(numBytes+size_);
        }

        writeData(src,numBytes,size_,srcBufferType,srcDeviceId);
        size_ += numBytes;
        // Do we worry about dirty flags here or does append avoid them
    }

    int8_t* Buffer::getMemoryPtr() {
        return mem_;
    }
    
}
