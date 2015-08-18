#ifndef FRAGMENTER_H
#define FRAGMENTER_H

#include <map>
#include <deque>
#include <list>
#include "../Shared/types.h"
#include "../Shared/mapd_shared_mutex.h"
#include "../DataMgr/ChunkMetadata.h"
#include "../Catalog/ColumnDescriptor.h"

namespace Data_Namespace {
class AbstractBuffer;
}

namespace Fragmenter_Namespace {

/**
 * @enum FragmenterType
 * stores the type of a child class of
 * AbstractTableFragmenter
 */

enum FragmenterType {
  INSERT_ORDER = 0  // these values persist in catalog.  make explicit
};

/**
 * @struct InsertData
 * @brief The data to be inserted using the fragment manager.
 *
 * The data being inserted is assumed to be in columnar format, and so the offset
 * to the beginning of each column can be calculated by multiplying the column size
 * by the number of rows.
 *
 * @todo support for variable-length data types
 */

struct InsertData {
  int databaseId;                  /// identifies the database into which the data is being inserted
  int tableId;                     /// identifies the table into which the data is being inserted
  std::vector<int> columnIds;      /// a vector of column ids for the row(s) being inserted
  size_t numRows;                  /// the number of rows being inserted
  std::vector<DataBlockPtr> data;  /// points to the start of the data block per column for the row(s) being inserted
};

/**
 * @struct FragmentInfo
 * @brief Used by Fragmenter classes to store info about each
 * fragment - the fragment id and number of tuples(rows)
 * currently stored by that fragment
 */

struct FragmentInfo {
  // std::vector<int>fragmentKeys;
  int fragmentId;
  size_t numTuples;
  size_t shadowNumTuples;
  std::vector<int> deviceIds;
  std::map<int, ChunkMetadata> chunkMetadataMap;
  std::map<int, ChunkMetadata> shadowChunkMetadataMap;

  FragmentInfo() : fragmentId(-1), numTuples(0), shadowNumTuples(0) {}
};

/**
 * @struct QueryInfo
 * @brief returned by Fragmenter classes in
 * getFragmentsForQuery - tells Executor which
 * fragments to scan from which fragmenter
 * (fragmenter id and fragment id needed for building
 * ChunkKey)
 */

struct QueryInfo {
  std::vector<int> chunkKeyPrefix;
  std::deque<FragmentInfo> fragments;
  size_t numTuples;
  mapd_shared_mutex* tableMutex;
  mapd_shared_lock<mapd_shared_mutex> tableLock;
  QueryInfo() : tableMutex(0) {}
  QueryInfo(mapd_shared_mutex* tableMutexIn) : tableMutex(tableMutexIn), tableLock(*tableMutex) {}
};

}  // Fragmenter_Namespace

#endif  // FRAGMENTER_H
