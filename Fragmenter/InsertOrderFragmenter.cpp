#include "InsertOrderFragmenter.h"
#include "../DataMgr/DataMgr.h"
#include "../DataMgr/AbstractBuffer.h"
#include <math.h>
#include <iostream>
#include <thread>

#include <assert.h>
#include <boost/lexical_cast.hpp>

#define DROP_FRAGMENT_FACTOR 0.9  // drop to 90% of max so we don't keep adding and dropping fragments

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;
using Chunk_NS::Chunk;

using namespace std;

namespace Fragmenter_Namespace {

InsertOrderFragmenter::InsertOrderFragmenter(const vector<int> chunkKeyPrefix,
                                             vector<Chunk>& chunkVec,
                                             Data_Namespace::DataMgr* dataMgr,
                                             const size_t maxFragmentRows,
                                             const size_t pageSize /*default 1MB*/,
                                             const size_t maxRows,
                                             const Data_Namespace::MemoryLevel defaultInsertLevel)
    : chunkKeyPrefix_(chunkKeyPrefix),
      dataMgr_(dataMgr),
      maxFragmentRows_(maxFragmentRows),
      pageSize_(pageSize),
      numTuples_(0),
      maxFragmentId_(-1),
      maxRows_(maxRows),
      fragmenterType_("insert_order"),
      defaultInsertLevel_(defaultInsertLevel),
      hasMaterializedRowId_(false) {
  // Note that Fragmenter is not passed virtual columns and so should only
  // find row id column if it is non virtual

  for (auto colIt = chunkVec.begin(); colIt != chunkVec.end(); ++colIt) {
    int columnId = colIt->get_column_desc()->columnId;
    columnMap_[columnId] = *colIt;
    if (colIt->get_column_desc()->columnName == "rowid") {
      hasMaterializedRowId_ = true;
      rowIdColId_ = columnId;
    }
  }
  getChunkMetadata();
}

InsertOrderFragmenter::~InsertOrderFragmenter() {
}

void InsertOrderFragmenter::getChunkMetadata() {
  std::vector<std::pair<ChunkKey, ChunkMetadata>> chunkMetadataVec;
  dataMgr_->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, chunkKeyPrefix_);
  // dataMgr_->getChunkMetadataVec(chunkMetadataVec);

  // data comes like this - database_id, table_id, column_id, fragment_id
  // but lets sort by database_id, table_id, fragment_id, column_id

  int fragmentSubKey = 3;
  std::sort(chunkMetadataVec.begin(),
            chunkMetadataVec.end(),
            [&](const std::pair<ChunkKey, ChunkMetadata>& pair1,
                const std::pair<ChunkKey, ChunkMetadata>& pair2) { return pair1.first[3] < pair2.first[3]; });

  for (auto chunkIt = chunkMetadataVec.begin(); chunkIt != chunkMetadataVec.end(); ++chunkIt) {
    int curFragmentId = chunkIt->first[fragmentSubKey];

    if (fragmentInfoVec_.empty() || curFragmentId != fragmentInfoVec_.back().fragmentId) {
      maxFragmentId_ = curFragmentId;
      fragmentInfoVec_.push_back(FragmentInfo());
      fragmentInfoVec_.back().fragmentId = curFragmentId;
      fragmentInfoVec_.back().numTuples = chunkIt->second.numElements;
      numTuples_ += fragmentInfoVec_.back().numTuples;
      for (const auto levelSize : dataMgr_->levelSizes_) {
        fragmentInfoVec_.back().deviceIds.push_back(curFragmentId % levelSize);
      }
      fragmentInfoVec_.back().shadowNumTuples = fragmentInfoVec_.back().numTuples;
    } else {
      if (chunkIt->second.numElements != fragmentInfoVec_.back().numTuples) {
        throw std::runtime_error("Inconsistency in num tuples within fragment");
      }
    }
    int columnId = chunkIt->first[2];
    fragmentInfoVec_.back().chunkMetadataMap[columnId] = chunkIt->second;
  }
  // Now need to get the insert buffers for each column - should be last
  // fragment
  if (fragmentInfoVec_.size() > 0) {
    int lastFragmentId = fragmentInfoVec_.back().fragmentId;
    int deviceId = fragmentInfoVec_.back().deviceIds[static_cast<int>(defaultInsertLevel_)];
    for (auto colIt = columnMap_.begin(); colIt != columnMap_.end(); ++colIt) {
      ChunkKey insertKey = chunkKeyPrefix_;  // database_id and table_id
      insertKey.push_back(colIt->first);    // column id
      insertKey.push_back(lastFragmentId);  // fragment id
      colIt->second.getChunkBuffer(dataMgr_, insertKey, defaultInsertLevel_, deviceId);
    }
  }
}

void InsertOrderFragmenter::dropFragmentsToSize(const size_t maxRows) {
  // not safe to call from outside insertData
  // b/c depends on insertLock around numTuples_

  if (numTuples_ > maxRows) {
    vector<int> dropFragIds;
    size_t targetRows = maxRows * DROP_FRAGMENT_FACTOR;
    while (numTuples_ > targetRows) {
      assert(fragmentInfoVec_.size() > 0);
      size_t numFragTuples = fragmentInfoVec_[0].numTuples;
      dropFragIds.push_back(fragmentInfoVec_[0].fragmentId);
      fragmentInfoVec_.pop_front();
      assert(numTuples_ >= numFragTuples);
      numTuples_ -= numFragTuples;
    }
    deleteFragments(dropFragIds);
  }
}

void InsertOrderFragmenter::deleteFragments(const vector<int>& dropFragIds) {
  mapd_unique_lock<mapd_shared_mutex> tableLock(tableMutex_);
  for (const auto fragId : dropFragIds) {
    for (const auto& col : columnMap_) {
      int colId = col.first;
      vector<int> fragPrefix = chunkKeyPrefix_;
      fragPrefix.push_back(colId);
      fragPrefix.push_back(fragId);
      dataMgr_->deleteChunksWithPrefix(fragPrefix);
    }
  }
}

void InsertOrderFragmenter::insertData(const InsertData& insertDataStruct) {
  mapd_lock_guard<std::mutex> insertLock(
      insertMutex_);  // prevent two threads from trying to insert into the same table simultaneously

  size_t numRowsLeft = insertDataStruct.numRows;
  size_t numRowsInserted = 0;
  vector<DataBlockPtr> dataCopy =
      insertDataStruct.data;  // bc append data will move ptr forward and this violates constness of InsertData
  if (numRowsLeft <= 0) {
    return;
  }

  FragmentInfo* currentFragment = 0;

  if (fragmentInfoVec_.empty()) {  // if no fragments exist for table
    currentFragment = createNewFragment(defaultInsertLevel_);
  } else {
    currentFragment = &(fragmentInfoVec_.back());
  }
  size_t startFragment = fragmentInfoVec_.size() - 1;

  while (numRowsLeft > 0) {  // may have to create multiple fragments for bulk insert
    // loop until done inserting all rows
    size_t rowsLeftInCurrentFragment = maxFragmentRows_ - currentFragment->shadowNumTuples;
    if (rowsLeftInCurrentFragment == 0) {
      currentFragment = createNewFragment();
      rowsLeftInCurrentFragment = maxFragmentRows_;
    }
    size_t numRowsToInsert = min(rowsLeftInCurrentFragment, numRowsLeft);
    // for each column, append the data in the appropriate insert buffer
    for (size_t i = 0; i < insertDataStruct.columnIds.size(); ++i) {
      int columnId = insertDataStruct.columnIds[i];
      auto colMapIt = columnMap_.find(columnId);
      assert(colMapIt != columnMap_.end());
      currentFragment->shadowChunkMetadataMap[columnId] =
          colMapIt->second.appendData(dataCopy[i], numRowsToInsert, numRowsInserted);
    }
    if (hasMaterializedRowId_) {
      size_t startId = maxFragmentRows_ * currentFragment->fragmentId + currentFragment->shadowNumTuples;
      int64_t* rowIdData = new int64_t[numRowsToInsert];
      for (size_t i = 0; i < numRowsToInsert; ++i) {
        rowIdData[i] = i + startId;
      }
      DataBlockPtr rowIdBlock;
      rowIdBlock.numbersPtr = reinterpret_cast<int8_t*>(rowIdData);
      auto colMapIt = columnMap_.find(rowIdColId_);
      currentFragment->shadowChunkMetadataMap[rowIdColId_] =
          colMapIt->second.appendData(rowIdBlock, numRowsToInsert, numRowsInserted);
      delete[] rowIdData;
    }

    currentFragment->shadowNumTuples = fragmentInfoVec_.back().numTuples + numRowsToInsert;
    numRowsLeft -= numRowsToInsert;
    numRowsInserted += numRowsToInsert;
  }
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  for (auto partIt = fragmentInfoVec_.begin() + startFragment; partIt != fragmentInfoVec_.end(); ++partIt) {
    partIt->numTuples = partIt->shadowNumTuples;
    partIt->chunkMetadataMap = partIt->shadowChunkMetadataMap;
  }
  numTuples_ += insertDataStruct.numRows;
  dropFragmentsToSize(maxRows_);
  // dataMgr_->checkpoint(); leave to upper layer to call checkpoint
}

FragmentInfo* InsertOrderFragmenter::createNewFragment(const Data_Namespace::MemoryLevel memoryLevel) {
  // also sets the new fragment as the insertBuffer for each column

  maxFragmentId_++;
  FragmentInfo newFragmentInfo;
  newFragmentInfo.fragmentId = maxFragmentId_;
  newFragmentInfo.shadowNumTuples = 0;
  newFragmentInfo.numTuples = 0;
  for (const auto levelSize : dataMgr_->levelSizes_) {
    newFragmentInfo.deviceIds.push_back(newFragmentInfo.fragmentId % levelSize);
  }

  for (map<int, Chunk>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
    // colMapIt->second.unpin_buffer();
    ChunkKey chunkKey = chunkKeyPrefix_;
    chunkKey.push_back(colMapIt->second.get_column_desc()->columnId);
    chunkKey.push_back(maxFragmentId_);
    colMapIt->second.createChunkBuffer(
        dataMgr_, chunkKey, memoryLevel, newFragmentInfo.deviceIds[static_cast<int>(memoryLevel)]);
    colMapIt->second.init_encoder();
  }

  mapd_lock_guard<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  fragmentInfoVec_.push_back(newFragmentInfo);
  return &(fragmentInfoVec_.back());
}

QueryInfo InsertOrderFragmenter::getFragmentsForQuery() {
  mapd_shared_lock<mapd_shared_mutex> readLock(fragmentInfoMutex_);
  QueryInfo queryInfo(&tableMutex_);
  queryInfo.chunkKeyPrefix = chunkKeyPrefix_;
  // right now we don't test predicate, so just return (copy of) all fragments
  queryInfo.fragments = fragmentInfoVec_;  // makes a copy
  readLock.unlock();
  queryInfo.numTuples = 0;
  auto partIt = queryInfo.fragments.begin();
  while (partIt != queryInfo.fragments.end()) {
    if (partIt->numTuples == 0) {
      // this means that a concurrent insert query inserted tuples into a new fragment but when the query came in we
      // didn't have this fragment.
      // To make sure we don't mess up the executor we delete this
      // fragment from the metadatamap (fixes earlier bug found
      // 2015-05-08)
      partIt = queryInfo.fragments.erase(partIt);
    } else {
      queryInfo.numTuples += partIt->numTuples;
      ++partIt;
    }
  }
  return queryInfo;
}

void InsertOrderFragmenter::getInsertBufferChunks() {
  for (map<int, Chunk>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
    // colMapIt->second.unpin_buffer();
    ChunkKey chunkKey = {fragmenterId_, maxFragmentId_, colMapIt->second.get_column_desc()->columnId};
    colMapIt->second.getChunkBuffer(dataMgr_, chunkKey, defaultInsertLevel_);
  }
}

}  // Fragmenter_Namespace
