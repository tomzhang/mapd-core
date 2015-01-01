/**
 * @file    BufferMgr.cpp
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#include "BufferMgr.h"
#include "Buffer.h"

#include <cassert>
#include <limits>

using namespace std;


namespace Buffer_Namespace {

    /// Allocates memSize bytes for the buffer pool and initializes the free memory map.
    BufferMgr::BufferMgr(const size_t maxBufferSize, const size_t slabSize, const size_t pageSize, AbstractDataMgr *parentMgr): maxBufferSize_(maxBufferSize), slabSize_(slabSize), pageSize_(pageSize), parentMgr_(parentMgr), bufferEpoch_(0) {
        assert(maxBufferSize_ > 0 && slabSize_ > 0 && pageSize_ > 0 && slabSize_ % pageSize_ == 0);
        numPagesPerSlab_ = slabSize_ / pageSize_; 
        maxNumSlabs_ = (maxBufferSize_/slabSize_);
        //addSlab(slabSize_);
    }

    /// Frees the heap-allocated buffer pool memory
    BufferMgr::~BufferMgr() {
        clear();
    }

    void BufferMgr::clear() {
        for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
            delete chunkIt -> second -> buffer;
        }
        chunkIndex_.clear();
        size_t numBufferSlabs = slabSegments_.size();
        slabs_.clear();
        slabSegments_.clear();
        //addSlab(slabSize_);
        unsizedSegs_.clear();
        bufferEpoch_ = 0;
    }
    
    /// Throws a runtime_error if the Chunk already exists
    AbstractBuffer * BufferMgr::createChunk(const ChunkKey &chunkKey, const mapd_size_t chunkPageSize, const mapd_size_t initialSize) {
        assert (chunkPageSize == pageSize_);
        // ChunkPageSize here is just for recording dirty pages
        if (chunkIndex_.find(chunkKey) != chunkIndex_.end()) {
            throw std::runtime_error("Chunk already exists");
        }
        BufferSeg bufferSeg(BufferSeg(-1,0,USED));
        bufferSeg.chunkKey = chunkKey;
        unsizedSegs_.push_back(bufferSeg);
        chunkIndex_[chunkKey] = std::prev(unsizedSegs_.end(),1); // need to do this before allocating Buffer because doing so could change the segment used
        allocateBuffer(chunkIndex_[chunkKey],chunkPageSize,initialSize); 
        //slabSegments_.back().buffer =  new Buffer(this, chunkKey, std::prev(slabSegments_.end(),1), chunkPageSize, initialSize); 
        //new Buffer(this, chunkIndex_[chunkKey], chunkPageSize, initialSize); // this line is admittedly a bit weird but the segment iterator passed into buffer takes the address of the new Buffer in its buffer member

        return chunkIndex_[chunkKey] -> buffer;
    }

    BufferList::iterator BufferMgr::evict(BufferList::iterator &evictStart, const size_t numPagesRequested, const int slabNum) {
        // We can assume here that buffer for evictStart either doesn't exist
        // (evictStart is first buffer) or was not free, so don't need ot merge
        // it
        auto evictIt = evictStart;
        size_t numPages = 0;
        size_t startPage = evictStart -> startPage;
        while (numPages < numPagesRequested) {
            assert (evictIt -> pinCount == 0);
            numPages += evictIt -> numPages;
            if (evictIt -> memStatus == USED && evictIt -> chunkKey.size() > 0) {
                chunkIndex_.erase(evictIt -> chunkKey);
            }
            evictIt = slabSegments_[slabNum].erase(evictIt); // erase operations returns next iterator - safe if we ever move to a vector (as opposed to erase(evictIt++)
        }
        BufferSeg dataSeg(startPage,numPagesRequested,USED,bufferEpoch_++); // until we can 
        dataSeg.pinCount++;
        dataSeg.slabNum = slabNum;
        auto dataSegIt = slabSegments_[slabNum].insert(evictIt,dataSeg); // Will insert before evictIt
        if (numPagesRequested < numPages) {
            size_t excessPages = numPages - numPagesRequested;
            if (evictIt != slabSegments_[slabNum].end() && evictIt -> memStatus == FREE) { // need to merge with current page
                evictIt -> startPage = startPage + numPagesRequested;
                evictIt -> numPages += excessPages;
            }
            else { // need to insert a free seg before evictIt for excessPages
                BufferSeg freeSeg(startPage + numPagesRequested,excessPages,FREE);
                slabSegments_[slabNum].insert(evictIt,freeSeg);
            }
        }
        return dataSegIt;
    }

    BufferList::iterator BufferMgr::reserveBuffer(BufferList::iterator &segIt, const size_t numBytes) {
        // doesn't resize to be smaller - like std::reserve
        //cout << "Reserve number bytes: " << numBytes << endl;
        size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
        //cout << "Reserve actual number bytes: " << numPagesRequested * pageSize_ << endl;
        size_t numPagesExtraNeeded = numPagesRequested -  segIt -> numPages;
       
        //cout << "Num extra pages needed: " << numPagesExtraNeeded << endl;

        if (numPagesRequested < segIt -> numPages) { // We already have enough pages in existing segment
            return segIt;
        }
        // First check for freeSeg after segIt 
        int slabNum = segIt -> slabNum;
        if (slabNum >= 0) { //not dummy page
            BufferList::iterator nextIt = std::next(segIt);
            if (nextIt != slabSegments_[slabNum].end() && nextIt -> memStatus == FREE && nextIt -> numPages >= numPagesExtraNeeded) { // Then we can just use the next BufferSeg which happens to be free
                size_t leftoverPages = nextIt -> numPages - numPagesExtraNeeded;
                segIt -> numPages = numPagesRequested;
                nextIt -> numPages = leftoverPages;
                nextIt -> startPage = segIt -> startPage + segIt -> numPages;
                return segIt;
            }
        }
        /* If we're here then we couldn't keep
         * buffer in existing slot - need to find
         * new segment, copy data over, and then
         * delete old
         */
        
        segIt -> pinCount++; // so we don't evict this while trying to find a new segment for it - @todo - maybe should go up top?
        auto newSegIt = findFreeBuffer(numBytes);
        /* Below should be in copy constructor for BufferSeg?*/

        newSegIt -> buffer = segIt -> buffer;
        //newSegIt -> buffer -> segIt_ = newSegIt;
        newSegIt -> chunkKey = segIt -> chunkKey;

        newSegIt -> buffer -> mem_ = slabs_[newSegIt->slabNum] + newSegIt -> startPage * pageSize_;
        // now need to copy over memory
        // only do this if the old segment is valid (i.e. not new w/
        // unallocated buffer
        if (segIt -> startPage >= 0 && segIt -> buffer -> mem_ != 0)  {
            memcpy(newSegIt -> buffer -> mem_, segIt -> buffer -> mem_, newSegIt->buffer->size());
        }
        // Deincrement pin count to reverse effect above above
        removeSegment(segIt);
        /*
        if (segIt -> slabNum < 0) {
            unsizedSegs_.erase(segIt);
        }
        else {
            slabSegments_[segIt -> slabNum].erase(segIt);
            //@todo - need to delete chunks around it
        }
        */
        chunkIndex_[newSegIt -> chunkKey] = newSegIt; 
        newSegIt -> pinCount = 0;

        return newSegIt;
    }

    BufferList::iterator BufferMgr::findFreeBufferInSlab(const size_t slabNum, const size_t numPagesRequested) {
            for (auto bufferIt = slabSegments_[slabNum].begin(); bufferIt != slabSegments_[slabNum].end(); ++bufferIt) {
                if (bufferIt -> memStatus == FREE && bufferIt -> numPages >= numPagesRequested) {
                    // startPage doesn't change
                    size_t excessPages = bufferIt -> numPages - numPagesRequested;
                    bufferIt -> numPages = numPagesRequested;
                    bufferIt -> memStatus = USED;
                    bufferIt -> pinCount = 1;
                    bufferIt -> lastTouched  = bufferEpoch_++;
                    bufferIt -> slabNum = slabNum;
                    if (excessPages > 0) {
                        BufferSeg freeSeg(bufferIt->startPage+numPagesRequested,excessPages,FREE);
                        auto tempIt = bufferIt; // this should make a copy and not be a reference
                        // - as we do not want to increment bufferIt
                        tempIt++;
                        slabSegments_[slabNum].insert(tempIt,freeSeg);
                    }
                    //std::cout << "Find free bufferIt: " << std::endl;
                    //printSeg(bufferIt);
                    return bufferIt;
                }
            }
            // If here then we did not find a free buffer of
            // sufficient size in
            // this slab, return the end iterator
            return slabSegments_[slabNum].end();
    }

    BufferList::iterator BufferMgr::findFreeBuffer(size_t numBytes) {
        size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
        if (numPagesRequested > numPagesPerSlab_) {
            throw std::runtime_error("Requtested memory allocation larger than slab size.");
        }
        
        size_t numSlabs = slabSegments_.size();

        for (size_t slabNum = 0; slabNum != numSlabs; ++slabNum) {
            auto segIt = findFreeBufferInSlab(slabNum,numPagesRequested);
            if (segIt != slabSegments_[slabNum].end()) {
                return segIt;
            }
        }

        // If we're here then we didn't find a free segment of sufficient size
        // First we see if we can add another slab
        if (numSlabs < maxNumSlabs_) {
            addSlab(slabSize_);
            return findFreeBufferInSlab(numSlabs, numPagesRequested); // has to return a free slab as long as requested buffer is smaller than the size of a slab
        }
        
        // If here then we can't add a slab - so we need to evict
        
        unsigned int minScore = std::numeric_limits<unsigned int>::max(); 
        // We're going for lowest score here, like golf
        // This is because score is the sum of the lastTouched score for all
        // pages evicted. Evicting less pages and older pages will lower the
        // score
        BufferList::iterator bestEvictionStart = slabSegments_[0].end();
        int slabNum = 0;

        for (auto slabIt = slabSegments_.begin(); slabIt != slabSegments_.end(); ++slabIt, ++slabNum) {
            for (auto bufferIt = slabIt -> begin(); bufferIt != slabIt -> end(); ++bufferIt) {
                /* Note there are some shortcuts we could take here - like we
                 * should never consider a USED buffer coming after a free buffer
                 * as we would have used the FREE buffer, but we won't worry about
                 * this for now
                 */

                // We can't evict pinned  buffers - only normal used
                // buffers

                if (bufferIt -> pinCount == 0) {
                    size_t pageCount = 0;
                    size_t score = 0;
                    bool solutionFound = false;
                    auto evictIt = bufferIt;
                    for (; evictIt != slabSegments_[slabNum].end(); ++evictIt) {
                       if (evictIt -> pinCount > 0) { // If pinned then we're at a dead end
                           break;
                        }
                        pageCount += evictIt -> numPages;
                        if (evictIt -> memStatus == USED) {
                            score += evictIt -> lastTouched;
                        }
                        if (pageCount >= numPagesRequested) {
                            solutionFound = true;
                            break;
                        }
                    }
                    if (solutionFound && score < minScore) {
                        minScore = score;
                        bestEvictionStart = bufferIt;
                    }
                    else if (evictIt == slabSegments_[slabNum].end()) {
                        // this means that every segment after this will fail as
                        // well, so our search has proven futile
                        //throw std::runtime_error ("Couldn't evict chunks to get free space");
                        break;
                        // in reality we should try to rearrange the buffer to get
                        // more contiguous free space
                    }
                    // other possibility is ending at PINNED - do nothing in this
                    // case

                }
            }
        }
        if (bestEvictionStart == slabSegments_[0].end()) {
            throw std::runtime_error ("Couldn't evict chunks to get free space");
        }
        bestEvictionStart = evict(bestEvictionStart,numPagesRequested,slabNum);
        return bestEvictionStart;
    }

    void BufferMgr::printSeg(BufferList::iterator &segIt) {
            std::cout << "Start page: " << segIt -> startPage << std::endl;
            std::cout << "Num Pages: " << segIt -> numPages << std::endl;
            std::cout << "Last touched: " << segIt -> lastTouched << std::endl;
            std::cout << "Pin count: " << segIt -> pinCount << std::endl;
            if (segIt -> memStatus == FREE)
                std::cout << "FREE" << std::endl;
            else {
                std::cout << "USED - Chunk: ";
                for (auto vecIt = segIt -> chunkKey.begin(); vecIt != segIt -> chunkKey.end(); ++vecIt) {
                    std::cout << *vecIt << ",";
                }
                std::cout << endl;
            }
    }




    void BufferMgr::printSegs() {
        int segNum = 1;
        int slabNum = 1;
        std::cout << std::endl << std::endl;
        for (auto slabIt = slabSegments_.begin(); slabIt != slabSegments_.end(); ++slabIt,++slabNum) {
            std::cout << "Slab Num: " << slabNum << std::endl;
            for (auto segIt = slabIt -> begin(); segIt != slabIt -> end(); ++segIt,++segNum) {
                std::cout << "Segment: " << segNum << std::endl;
                printSeg(segIt);
                std::cout << std::endl;
            }
            std::cout << "--------------------" << std::endl;
        }
    }

    void BufferMgr::printMap() {
        int segNum = 1;
        std::cout << std::endl << "Map Contents: " <<  std::endl;
        for (auto segIt = chunkIndex_.begin(); segIt != chunkIndex_.end(); ++segIt,++segNum) {
            std::cout << "Chunk " << segNum << ": ";
            for (auto vecIt = segIt -> first.begin(); vecIt != segIt -> first.end(); ++vecIt) {
                std::cout << *vecIt << ",";
            }
            std::cout << std::endl;
            printSeg(segIt -> second);
        }
        std::cout << "--------------------" << std::endl;
    }


    
    /// This method throws a runtime_error when deleting a Chunk that does not exist.
    void BufferMgr::deleteChunk(const ChunkKey &key) {
        // lookup the buffer for the Chunk in chunkIndex_
        auto chunkIt = chunkIndex_.find(key);
        //Buffer *buffer = chunkIt -> second -> buffer;

        if (chunkIndex_.find(key) == chunkIndex_.end()) {
            throw std::runtime_error("Chunk does not exist");
        }
        auto  segIt = chunkIt->second;
        delete segIt->buffer; // Delete Buffer for segment
        removeSegment(segIt);
        chunkIndex_.erase(chunkIt);
    }



     void BufferMgr::removeSegment(BufferList::iterator &segIt) {
         // Note: does not delete buffer as this may be moved somewhere else
        int slabNum = segIt -> slabNum;
        if (slabNum < 0) {
            unsizedSegs_.erase(segIt);
        }
        else {
            if (segIt != slabSegments_[slabNum].begin()) {
                auto prevIt = std::prev(segIt);
                //std::cout << "PrevIt: " << std::endl;
                //printSeg(prevIt);
                if (prevIt -> memStatus == FREE) { 
                    segIt -> startPage = prevIt -> startPage;
                    segIt -> numPages += prevIt -> numPages; 
                    slabSegments_[slabNum].erase(prevIt);
                }
            }
            auto nextIt = std::next(segIt);
            if (nextIt != slabSegments_[slabNum].end()) {
                if (nextIt -> memStatus == FREE) { 
                    segIt -> numPages += nextIt -> numPages;
                    slabSegments_[slabNum].erase(nextIt);
                }
            }
            segIt -> memStatus = FREE;
            segIt -> pinCount = 0;
            segIt -> buffer = 0;
        }

        /*Below is the other, more complicted algorithm originally chosen*/

        //size_t numPageInSeg = segIt -> numPages;
        //bool prevFree = false; 
        //bool nextFree = false; 
        //if (segIt != slabSegments_.begin()) {
        //    auto prevIt = std::prev(segIt);
        //    if (prevIt -> memStatus == FREE) { 
        //        prevFree = true;
        //        prevIt -> numPages += numPagesInSeg;
        //        segIt = prevIt;
        //    }
        //}
        //auto nextIt = std::next(segIt);
        //if (nextIt != slabSegments_.end()) {
        //    if (nextIt -> memStatus == FREE) {
        //        nextFree = true;
        //        if (prevFree) {
        //            prevIt -> numPages += nextIt -> numPages; 
        //            slabSegments_.erase(nextIt);
        //        }
        //        else {
        //            nextIt -> numPages += numPagesInSeg;
        //        }
        //    }
        //}
        //if (!prevFree && !nextFree) {
        //    segIt -> memStatus = FREE;
        //    segIt -> pinCount = 0;
        //    segIt -> buffer = 0;
        //}
        //else {
        //    slabSegments_.erase(segIt);
        //}
    }

    void BufferMgr::checkpoint() {
        for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
            if (chunkIt -> second -> buffer -> isDirty_) {
                cout << "Flushing: ";
                for (auto vecIt = chunkIt -> second -> chunkKey.begin(); vecIt != chunkIt -> second -> chunkKey.end(); ++vecIt) {
                    std::cout << *vecIt << ",";
                }
                std::cout << std::endl;
                parentMgr_ -> putChunk(chunkIt -> second -> chunkKey, chunkIt -> second -> buffer); 
                chunkIt -> second -> buffer -> clearDirtyBits();
            }
            //parentMgr_ -> checkpoint();
        }
    }
    
    /// Returns a pointer to the Buffer holding the chunk, if it exists; otherwise,
    /// throws a runtime_error.
    AbstractBuffer* BufferMgr::getChunk(ChunkKey &key, const mapd_size_t numBytes) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt != chunkIndex_.end()) {
            chunkIt -> second -> pinCount++; 
            chunkIt -> second -> lastTouched = bufferEpoch_++;
            return chunkIt -> second -> buffer; 
        }
        else { // If wasn't in pool then we need to fetch it
            AbstractBuffer * buffer = createChunk(key,pageSize_,numBytes);
            cout << "About to fetchChunk" << endl;
            parentMgr_ -> fetchChunk(key,buffer,numBytes); // this should put buffer in a BufferSegment
            return buffer;
        }
    }

    //void BufferMgr::getChunks(

    void BufferMgr::fetchChunk(const ChunkKey &key, AbstractBuffer *destBuffer, const mapd_size_t numBytes) {
        throw std::runtime_error("Not implemented");
    }
    
    AbstractBuffer* BufferMgr::putChunk(const ChunkKey &key, AbstractBuffer *d, const mapd_size_t numBytes) {
        //assert(d->size() > 0);
        //mapd_size_t nbytes = d->size();
        //// create Chunk if it doesn't exist
        //auto chunkPageSizeIt = chunkPageSize_.find(key);
        //if (chunkPageSizeIt == chunkPageSize_.end())
        //    createChunk(key, d->pageSize());
        //
        //// check if Chunk's Buffer exists
        //Buffer *b = nullptr;
        //auto chunkIt = chunkIndex_.find(key);
        //if (chunkIndex_.find(key) == chunkIndex_.end()) {
        //    b = new Buffer(nullptr, d->pageCount(), d->pageSize(), -1);
        //    chunkIndex_.insert(std::pair<ChunkKey, Buffer*>(key, b));
        //}
        //else {
        //    b = chunkIt->second;
        //}
        //
        //// should be a consistent page size for a given ChunkKey
        //assert(b->pageSize() == d->pageSize());
        //
        //// if necessary, reserve memory for b
        //if (b->mem_ == nullptr) {

        //    // Find n bytes of free memory in the buffer pool
        //    auto freeMemIt = freeMem_.lower_bound(nbytes);
        //    if (freeMemIt == freeMem_.end()) {
        //        delete b;
        //        chunkIndex_.erase(chunkIt);
        //        throw std::runtime_error("Out of memory");
        //        // @todo eviction strategies
        //    }
        //    
        //    // Save free memory information
        //    mapd_size_t freeMemSize = freeMemIt->first;
        //    mapd_addr_t bufAddr = freeMemIt->second;
        //    
        //    // update Buffer's pointer
        //    b->mem_ = bufAddr;
        //    
        //    // Remove entry from map, and insert new entry
        //    freeMem_.erase(freeMemIt);
        //    if (freeMemSize - nbytes > 0)
        //        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(freeMemSize - nbytes, bufAddr + nbytes));
        //}
        //
        //// b and d should be the same size
        //if (b->size() != d->size())
        //    throw std::runtime_error("Size mismatch between source and destination buffers.");

        //// read the contents of d into b
        //d->read(b->mem_, 0);
        //
        //return b;
    }
    
    /// client is responsible for deleting memory allocated for b->mem_
    AbstractBuffer* BufferMgr::createBuffer(mapd_size_t pageSize, mapd_size_t nbytes) {
        //assert(pageSize > 0 && nbytes > 0);
        //mapd_size_t numPages = (pageSize + nbytes - 1) / pageSize;
        //Buffer *b = new Buffer(nullptr, numPages, pageSize, -1);
        //
        //// Find nbytes of free memory in the buffer pool
        //auto freeMemIt = freeMem_.lower_bound(nbytes);
        //if (freeMemIt == freeMem_.end()) {
        //    delete b;
        //    throw std::runtime_error("Out of memory");
        //    // @todo eviction strategies
        //}
        //
        //// Save free memory information
        //mapd_size_t freeMemSize = freeMemIt->first;
        //mapd_addr_t bufAddr = freeMemIt->second;

        //// update Buffer's pointer
        //b->mem_ = bufAddr;
        //
        //// Remove entry from map, and insert new entry
        //freeMem_.erase(freeMemIt);
        //if (freeMemSize - nbytes > 0)
        //    freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(freeMemSize - nbytes, bufAddr + nbytes));
        //
        //return b;
    }
    
    void BufferMgr::deleteBuffer(AbstractBuffer *d) {
        //assert(d);
        //Buffer *b = (Buffer*)d;
        //
        //// return the free memory used by the Chunk back to the free memory pool
        //freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(b->size(), b->mem_));
        //delete[] b->mem_;
        //delete b;
    }
    
    AbstractBuffer* BufferMgr::putBuffer(AbstractBuffer *d) {
        // @todo write this putBuffer() method
        return nullptr;
    }
    
    mapd_size_t BufferMgr::size() {
        return slabs_.size()*pageSize_*numPagesPerSlab_;
    }
    
}
