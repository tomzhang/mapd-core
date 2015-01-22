#include "LinearTablePartitioner.h"
#include "../../DataMgr/DataMgr.h"
#include "../../DataMgr/AbstractBuffer.h"
#include <math.h>
#include <iostream>
#include <thread>

#include <assert.h>
#include <boost/lexical_cast.hpp>

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;

using namespace std;

namespace Partitioner_Namespace {

LinearTablePartitioner::LinearTablePartitioner(const vector <int> chunkKeyPrefix, vector <ColumnInfo> &columnInfoVec, Data_Namespace::DataMgr *dataMgr, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize /*default 1MB*/) :
		chunkKeyPrefix_(chunkKeyPrefix), dataMgr_(dataMgr), maxPartitionRows_(maxPartitionRows), pageSize_(pageSize), maxPartitionId_(-1), partitionerType_("insert_order"){
    for (auto colIt = columnInfoVec.begin(); colIt != columnInfoVec.end(); ++colIt) {
        columnMap_[colIt -> columnId] = *colIt; 
    }
}

LinearTablePartitioner::~LinearTablePartitioner() {

}

void LinearTablePartitioner::insertData (const InsertData &insertDataStruct) {
    mapd_size_t numRowsLeft = insertDataStruct.numRows;
    mapd_size_t numRowsInserted = 0;
    vector <PartitionInfo *> partitionsToBeUpdated;  
    if (maxPartitionId_ < 0 && numRowsLeft > 0) { // if no partitions exist for table and there is >= 1 row to insert
        createNewPartition();
    }
    while (numRowsLeft > 0) { // may have to create multiple partitions for bulk insert
        // loop until done inserting all rows
        mapd_size_t rowsLeftInCurrentPartition = maxPartitionRows_ - partitionInfoVec_.back().numTuples;
        if (rowsLeftInCurrentPartition == 0) {
            createNewPartition();
            rowsLeftInCurrentPartition = maxPartitionRows_;
        }
        mapd_size_t numRowsToInsert = min(rowsLeftInCurrentPartition, numRowsLeft);
        // for each column, append the data in the appropriate insert buffer
        partitionsToBeUpdated.push_back(&(partitionInfoVec_.back()));
        for (int i = 0; i < insertDataStruct.columnIds.size(); ++i) {
            int columnId = insertDataStruct.columnIds[i];
            auto colMapIt = columnMap_.find(columnId);
            assert(colMapIt != columnMap_.end());
            AbstractBuffer *insertBuffer = colMapIt->second.insertBuffer;
            partitionInfoVec_.back().shadowChunkMetadataMap[columnId] = colMapIt->second.insertBuffer->encoder->appendData(static_cast<mapd_addr_t>(insertDataStruct.data[i]),numRowsToInsert);
        }


        partitionInfoVec_.back().shadowNumTuples = partitionInfoVec_.back().numTuples + numRowsToInsert;
        numRowsLeft -= numRowsToInsert;
        numRowsInserted += numRowsToInsert;
    }
    boost::unique_lock < boost::shared_mutex > writeLock (readWriteMutex_);
    for (auto partIt = partitionsToBeUpdated.begin(); partIt != partitionsToBeUpdated.end(); ++partIt) {
        (*partIt)->numTuples = (*partIt)->shadowNumTuples;
        (*partIt)->chunkMetadataMap=(*partIt)->shadowChunkMetadataMap;
    }
}

void LinearTablePartitioner::createNewPartition() { 
    // also sets the new partition as the insertBuffer for each column

    // iterate through all ColumnInfo structs in map, unpin previous insert buffer and
    // create new insert buffer
    maxPartitionId_++;
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer != 0) {
            colMapIt -> second.insertBuffer -> unPin();
        }
        ChunkKey chunkKey =  chunkKeyPrefix_;
        chunkKey.push_back(colMapIt->second.columnId);
        chunkKey.push_back(maxPartitionId_);
        colMapIt->second.insertBuffer = dataMgr_->createChunk(Data_Namespace::DISK_LEVEL,chunkKey);
        colMapIt->second.insertBuffer->initEncoder(colMapIt->second.columnType,colMapIt->second.encodingType,colMapIt->second.encodingBits);
    }
    PartitionInfo newPartitionInfo;
    newPartitionInfo.partitionId = maxPartitionId_;
    newPartitionInfo.numTuples = 0; 

    boost::unique_lock < boost::shared_mutex > writeLock (readWriteMutex_);
    partitionInfoVec_.push_back(newPartitionInfo);
}

void LinearTablePartitioner::getPartitionsForQuery(QueryInfo &queryInfo) {
    queryInfo.chunkKeyPrefix = chunkKeyPrefix_;
    // right now we don't test predicate, so just return (copy of) all partitions 
    {
        boost::shared_lock < boost::shared_mutex > readLock (readWriteMutex_);
        queryInfo.partitions = partitionInfoVec_; //makes a copy
    }
}


void LinearTablePartitioner::getInsertBufferChunks() {
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        assert (colMapIt -> second.insertBuffer == NULL);
        if (colMapIt -> second.insertBuffer != NULL) {
            // should really always be null - we should just be using this
            // method up front
            colMapIt -> second.insertBuffer -> unPin();
        }
        ChunkKey chunkKey = {partitionerId_, maxPartitionId_,  colMapIt -> second.columnId};
        colMapIt -> second.insertBuffer = dataMgr_->getChunk(Data_Namespace::DISK_LEVEL,chunkKey);
    }
}
/*
void LinearTablePartitioner::readState() {
    string partitionQuery ("select fragment_id, num_rows from fragments where partitioner_id = " + boost::lexical_cast <string> (partitionerId_));
    partitionQuery += " order by fragment_id";
    mapd_err_t status = pgConnector_.query(partitionQuery);
    assert(status == MAPD_SUCCESS);
    size_t numRows = pgConnector_.getNumRows();
    partitionInfoVec_.resize(numRows);
    for (int r = 0; r < numRows; ++r) {
        partitionInfoVec_[r].partitionId = pgConnector_.getData<int>(r,0);  
        partitionInfoVec_[r].numTuples = pgConnector_.getData<int>(r,1);
    }
    if (numRows > 0) {
        maxPartitionId_ = partitionInfoVec_[numRows-1].partitionId; 
        getInsertBufferChunks();
    }
    string statsQuery ("select fragment_id");
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        int columnId = colMapIt -> first;
        string baseStatsColumnName = "col_" + boost::lexical_cast <string> (columnId);
        statsQuery += "," + baseStatsColumnName + "_min," + baseStatsColumnName + "_max";
    }
    statsQuery += " from partitioner_" + boost::lexical_cast <string> (partitionerId_) + "_stats";
    status = pgConnector_.query(statsQuery);
    assert(status == MAPD_SUCCESS);
    numRows = pgConnector_.getNumRows();
}
void LinearTablePartitioner::writeState() {
    // do we want this to be fully durable or will allow ourselves
    // to delete existing rows for this table
    // out of convenience before adding the
    // newest state back in?
    // Will do latter for now as we do not 
    // plan on using postgres forever for metadata
    if (isDirty_) {
         string deleteQuery ("delete from fragments where partitioner_id = " + boost::lexical_cast <string> (partitionerId_));
         mapd_err_t status = pgConnector_.query(deleteQuery);
         assert(status == MAPD_SUCCESS);
        for (auto partIt = partitionInfoVec_.begin(); partIt != partitionInfoVec_.end(); ++partIt) {
            string insertQuery("INSERT INTO fragments (partitioner_id, fragment_id, num_rows) VALUES (" + boost::lexical_cast<string>(partitionerId_) + "," + boost::lexical_cast<string>(partIt -> partitionId) + "," + boost::lexical_cast<string>(partIt -> numTuples) + ")"); 
            status = pgConnector_.query(insertQuery);
             assert(status == MAPD_SUCCESS);
        }
    }
    isDirty_ = false;
}
*/

} // Partitioner_Namespace
