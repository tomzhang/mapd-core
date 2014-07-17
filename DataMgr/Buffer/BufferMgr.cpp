/**
 * @file	BufferMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 */
#include <iostream>
#include <cassert>
#include <cstring>
#include <exception>
#include "BufferMgr.h"
#include "Buffer.h"
#include "../File/FileMgr.h"

using std::cout;
using std::endl;
using File_Namespace::Chunk;

namespace Buffer_Namespace {

BufferMgr::BufferMgr(mapd_size_t hostMemSize, FileMgr *fm) {
    assert(hostMemSize > 0);
    fm_ = fm;
    hostMemSize_ = hostMemSize;
    hostMem_ = new mapd_addr_t[hostMemSize];
    freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t*>(hostMemSize, hostMem_));

#ifdef DEBUG_VERBOSE
    printMemAlloc();
#endif
}

BufferMgr::~BufferMgr() {
    // Delete buffers
    while (buffers_.size() > 0) {
        delete buffers_.back();
        buffers_.pop_back();
    }

    // Delete host memory
    delete[] hostMem_;
}

Buffer* BufferMgr::createBuffer(mapd_size_t numPages, mapd_size_t pageSize) {
    // Compute total bytes needed
    mapd_size_t n = numPages * pageSize;
    assert(n > 0);

    // Find n bytes of free memory in the buffer pool
    auto it = freeMem_.lower_bound(n);
    if (it == freeMem_.end()) {
        fprintf(stderr, "[%s:%d] Error: unable to find %lu bytes available in the buffer pool.\n", __func__, __LINE__, n);
        // @todo eviction strategies
        return NULL;
    }

    // Save free memory information
    mapd_size_t freeMemSize = it->first;
    mapd_addr_t *bufAddr = it->second;

    // Remove entry from map, and insert new entry
    freeMem_.erase(it);
    if (freeMemSize - n > 0)
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t*>(freeMemSize - n, bufAddr + n));

    // Create Buffer object and add to the BufferMgr
    Buffer *b = new Buffer(bufAddr, numPages, pageSize);
    buffers_.push_back(b);

    return b;
}

void BufferMgr::deleteBuffer(Buffer *b) {
    // free pages
    // @todo free pages of buffer

    // free buffer and remove from BufferMgr's list of buffers
    delete b;
    buffers_.remove(b); // @todo thread safe needed?
}

/// Presently, only returns the Buffer if it is not currently pinned
Buffer* BufferMgr::getChunkBuffer(const ChunkKey &key) {
    Buffer *b;

    // Check if buffer is already cached
    b = findChunkBuffer(key);
    if (b && !b->pinned())
        return NULL;

    // Determine number of pages and page size for chunk
    int numPages;
    mapd_size_t size;
    if ((fm_->getChunkSize(key, &numPages, &size)) == MAPD_FAILURE)
        return NULL;
    assert((size % numPages) == 0);

    // Create buffer and load chunk
    b = createBuffer(numPages, size / numPages);
    if ((fm_->getChunk(key, b->host_ptr())) == NULL) {
        deleteBuffer(b);
        return NULL;
    }
    return b;
}

/// Presently, only returns the pointer if it the buffer is not currently pinned
mapd_addr_t* BufferMgr::getChunkAddr(const ChunkKey &key, mapd_size_t *length) {
    Buffer *b = findChunkBuffer(key);
    if (b && b->pinned())
        return NULL;
    else if (!b)
        return NULL;
    
    if (length) *length = b->length();
    b->pin();
    return b->host_ptr();
}

/// Presently, only flushes a chunk if it is unpinned, and flushes it right away (no queue)
bool BufferMgr::flushChunk(const ChunkKey &key) {
    Buffer *b = findChunkBuffer(key);
    if (b && b->pinned())
        return false;
    if ((fm_->putChunk(key, b->length(), b->host_ptr())) != MAPD_SUCCESS)
        return false;
    return true;
}

void BufferMgr::printMemAlloc() {
    mapd_size_t freeMemSize = 0;
    auto it = freeMem_.begin();
    for (; it != freeMem_.end(); ++it)
        freeMemSize += it->first;

    printf("Total memory  = %lu bytes\n", hostMemSize_);
    printf("Used memory   = %lu bytes\n", hostMemSize_ - freeMemSize);
    printf("Free memory   = %lu bytes\n", freeMemSize);
    printf("# of buffers  = %lu\n", buffers_.size());
}

Buffer* BufferMgr::findChunkBuffer(const ChunkKey key) {
    auto it = chunkIndex_.find(key);
    if (it == chunkIndex_.end()) // not found
        return NULL;
    return it->second;
}

} // Buffer_Namespace