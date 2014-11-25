/**
 * @file        FileBuffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "FileBuffer.h"
#include "File.h"
#include "FileMgr.h"
#include <map>

using namespace std;

FileBuffer::FileBuffer(mapd_size_t pageSize, FileMgr *fm) : pageSize_(pageSize), fm_(fm) {
    assert(fm_);
    // NOP
}

FileBuffer::~FileBuffer() {
    // need to free pages
    // NOP
}

void FileBuffer::read(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset) {
    // variable declarations
    mapd_addr_t curPtr = dst;    // a pointer to the current location in dst being written to
    mapd_size_t startPage = offset / pageSize_;
    mapd_size_t startPageOffset = offset % pageSize_;
    mapd_size_t numPagesToRead; = (numBytes + startPageOffset + pageSize_ - 1) / pageSize_;
    assert (startPage + numPagesToRead <= multiPages_.size());
    mapd_size_t bytesLeft = numBytes;

    // Traverse the logical pages
    for (size_t pageNum = startPage; pageNum < startPage  + numPagesToRead; ++pageNum) {

        assert(multiPages_[pageNum].pageSize == pageSize_);
        Page page = multiPages_[pageNum].current();
        printf("read: fileId=%d pageNum=%lu pageSize=%lu\n", page.fileId, page.pageNum, pageSize_);

        FILE *f = fm_ -> files_[page.fileId] -> f;
        assert(f);

        // Read the page into the destination (dst) buffer at its
        // current (cur) location
        size_t bytesRead;
        if (pageNum == startPage) {
            bytesRead = File_Namespace::read(f, page.pageNum * pageSize_ + startPageOffset, min(pageSize_ - startPageOffset,bytesLeft), curPtr);
        }
        else {
            bytesRead = File_Namespace::read(f, page.pageNum * pageSize_, min(pageSize_,bytesLeft), curPtr);
        }
        curPtr += bytesRead;
        bytesLeft -= bytesRead;
    }
    assert (bytesLeft == 0);
}

void FileBuffer::copyPage(Block srcPage, Block destPage, const mapd_size_t numBytes, const mapd_size_t offset) { 
    FILE *srcFile = fm_ -> files_[srcPage.fileId] -> f;
    FILE *destFile = fm_ -> files_[destPage.fileId] -> f;
    mapd_addr_t buffer = new mapd_byte_t [numBytes]; 
    size_t bytesRead = File_Namespace::read(srcFile,srcPage.pageNum * pageSize_ + offset, buffer);
    assert(bytesRead == numBytes);
    size_t bytesWritten = File_Namespace::write(destFile,destPage.pageNum * pageSize_ + offset,buffer);
    assert(bytesWritten == numBytes);
    delete [] buffer;
}


void FileBuffer::write(mapd_addr_t src,  const mapd_size_t numBytes, const mapd_size_t offset) {
    mapd_size_t startPage = offset / pageSize_;
    mapd_size_t startPageOffset = offset % pageSize_;
    mapd_size_t numPagesToWrite = (numBytes + startPageOffset + pageSize_ - 1) / pageSize_; 
    mapd_size_t bytesLeft = numBytes;
    mapd_addr_t curPtr = src;    // a pointer to the current location in dst being written to
    mapd_size_t initialNumPages = multiPages_size();
    int epoch = fm_-> epoch();

    for (size_t pageNum = startPage; pageNum < startPage  + numPagesToWrite; ++pageNum) {
        Block *page = nullptr;
        if (pageNum >= initialNumPages) {
            page = fm_ -> requestFreeBlock(pageSize_);
        }
        else if (multiPages_[pageNum].epochs.back() < epoch) { // need to create new page b/c this current one lags epoch and we can't overwrite it 
// also need to copy if we are on first or last page
            Block *lastPage = multiPages_[pageNum].current();
            page = fm_ -> requestFreeBlock(pageSize_);
            if (pageNum == startPage && startPageOffset > 0) {
                copyPage(lastPage,page,startOffset,0);
            }
            if (pageNum == startPage + numPagesToWrite && bytesLeft > 0) { // bytesLeft should always > 0
                copyPage(lastPage,page,pageSize_-bytesLeft,bytesLeft); // these would be empty if we're appending but we won't worry about it right now
            }
        }
        else {
            // we already have a new block at current
            // epoch for this page - just grab this page
            page = multiPages_[pageNum].current();
        }
        FILE *f = fm_ -> files_[page.fileId] -> f;
        size_t bytesWritten;
        if (pageNum == startPage) {
            bytesWritten = File_Namespace::write(f,page.pageNum*pageSize_ + startPageOffset, min (pageSize_ - startPageOffset,bytesLeft),curPtr);
        }
        else {
            bytesWritten = File_Namespace::write(f, page.pageNum * pageSize_, min(pageSize_,bytesLeft), curPtr);
        }
        curPtr += bytesWritten;
        bytesLeft -= bytesWritten;
    }
    assert (bytesLeft == 0);
}
