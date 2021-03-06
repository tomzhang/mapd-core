/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * @file    JoinHashTable.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_JOINHASHTABLE_H
#define QUERYENGINE_JOINHASHTABLE_H

#include "ExpressionRange.h"
#include "ColumnarResults.h"
#include "InputDescriptors.h"
#include "InputMetadata.h"
#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"
#include "../Chunk/Chunk.h"

#include <llvm/IR/Value.h>
#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <memory>
#include <mutex>
#include <stdexcept>

class Executor;

class HashJoinFail : public std::runtime_error {
 public:
  HashJoinFail(const std::string& reason) : std::runtime_error(reason) {}
};

class JoinHashTable {
 public:
  static std::shared_ptr<JoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
      const Catalog_Namespace::Catalog& cat,
      const std::vector<InputTableInfo>& query_infos,
      const std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      Executor* executor);

  int64_t getJoinHashBuffer(const ExecutorDeviceType device_type, const int device_id) noexcept {
#ifdef HAVE_CUDA
    if (device_type == ExecutorDeviceType::CPU) {
      CHECK(cpu_hash_table_buff_);
    } else {
      CHECK_LT(static_cast<size_t>(device_id), gpu_hash_table_buff_.size());
    }
    return device_type == ExecutorDeviceType::CPU ? reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0])
                                                  : gpu_hash_table_buff_[device_id];
#else
    CHECK(device_type == ExecutorDeviceType::CPU);
    CHECK(cpu_hash_table_buff_);
    return reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0]);
#endif
  }

 private:
  JoinHashTable(const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
                const Analyzer::ColumnVar* col_var,
                const Catalog_Namespace::Catalog& cat,
                const std::vector<InputTableInfo>& query_infos,
                const Data_Namespace::MemoryLevel memory_level,
                const ExpressionRange& col_range,
                Executor* executor)
      : qual_bin_oper_(qual_bin_oper),
        cat_(cat),
        query_infos_(query_infos),
        memory_level_(memory_level),
        col_range_(col_range),
        executor_(executor) {
    CHECK(col_range.getType() == ExpressionRangeType::Integer);
  }

  std::pair<const int8_t*, size_t> getColumnFragment(
      const Analyzer::ColumnVar& hash_col,
      const Fragmenter_Namespace::FragmentInfo& fragment,
      const Data_Namespace::MemoryLevel effective_mem_lvl,
      const int device_id,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      std::map<int, std::shared_ptr<const ColumnarResults>>& frags_owner);

  std::pair<const int8_t*, size_t> getAllColumnFragments(
      const Analyzer::ColumnVar& hash_col,
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      std::map<int, std::shared_ptr<const ColumnarResults>>& frags_owner);

  int reify(const int device_count);
  int initHashTableForDevice(const ChunkKey& chunk_key,
                             const int8_t* col_buff,
                             const size_t num_elements,
                             const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols,
                             const Data_Namespace::MemoryLevel effective_memory_level,
                             const int device_id);
  void initHashTableOnCpuFromCache(const ChunkKey& chunk_key,
                                   const size_t num_elements,
                                   const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols);
  void putHashTableOnCpuToCache(const ChunkKey& chunk_key,
                                const size_t num_elements,
                                const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols);
  int initHashTableOnCpu(const int8_t* col_buff,
                         const size_t num_elements,
                         const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols,
                         const int32_t hash_entry_count,
                         const int32_t hash_join_invalid_val);

  llvm::Value* codegenSlot(const CompilationOptions&) noexcept;

  const InputTableInfo& getInnerQueryInfo(const Analyzer::ColumnVar* inner_col);

  std::shared_ptr<Analyzer::BinOper> qual_bin_oper_;
  const Catalog_Namespace::Catalog& cat_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  std::shared_ptr<std::vector<int32_t>> cpu_hash_table_buff_;
  std::mutex cpu_hash_table_buff_mutex_;
#ifdef HAVE_CUDA
  std::vector<CUdeviceptr> gpu_hash_table_buff_;
#endif
  ExpressionRange col_range_;
  Executor* executor_;

  struct JoinHashTableCacheKey {
    const ExpressionRange col_range;
    const Analyzer::ColumnVar inner_col;
    const Analyzer::ColumnVar outer_col;
    const size_t num_elements;
    const ChunkKey chunk_key;

    bool operator==(const struct JoinHashTableCacheKey& that) const {
      return col_range == that.col_range && inner_col == that.inner_col && outer_col == that.outer_col &&
             num_elements == that.num_elements && chunk_key == that.chunk_key;
    }
  };

  static std::vector<std::pair<JoinHashTableCacheKey, std::shared_ptr<std::vector<int32_t>>>> join_hash_table_cache_;
  static std::mutex join_hash_table_cache_mutex_;

  static const int ERR_MULTI_FRAG{-2};
  static const int ERR_FAILED_TO_FETCH_COLUMN{-3};
  static const int ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN{-4};

  friend class Executor;
};

inline std::string get_table_name_by_id(const int table_id, const Catalog_Namespace::Catalog& cat) {
  if (table_id >= 1) {
    const auto td = cat.getMetadataForTable(table_id);
    CHECK(td);
    return td->tableName;
  }
  return "$TEMPORARY_TABLE" + std::to_string(-table_id);
}

#endif  // QUERYENGINE_JOINHASHTABLE_H
