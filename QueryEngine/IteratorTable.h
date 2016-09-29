/**
 * @file    IteratorTable.h
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Basic constructors and methods of the iterator table interface.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_ITERATORTABLE_H
#define QUERYENGINE_ITERATORTABLE_H

#include "ResultRows.h"

#include <boost/variant.hpp>

class QueryExecutionContext;

struct BufferFragment {
  int64_t* data;
  size_t row_count;
};

class IteratorTable {
 public:
  IteratorTable(const QueryMemoryDescriptor& query_mem_desc,
                const std::vector<Analyzer::Expr*>& targets,
                const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                int64_t* group_by_buffer,
                const size_t groups_buffer_entry_count,
                const std::vector<std::vector<const int8_t*>>& col_buffers,
                const ssize_t frag_id,
                const ExecutorDeviceType device_type,
                const int device_id);

  IteratorTable(const std::vector<TargetInfo>& targets,
                const QueryMemoryDescriptor& query_mem_desc,
                const ExecutorDeviceType device_type,
                const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  // Empty iterator table constructor
  IteratorTable();

  void append(const IteratorTable& that) {
    buffer_frags_.insert(buffer_frags_.end(), that.buffer_frags_.begin(), that.buffer_frags_.end());
  }

  void fetchLazy(const std::unordered_map<size_t, ssize_t>& lazy_col_local_ids,
                 const std::vector<std::vector<const int8_t*>>& col_buffers,
                 const ssize_t frag_id);

  size_t colCount() const { return just_explain_ ? 1 : query_mem_desc_.agg_col_widths.size(); }

  size_t fragCount() const { return buffer_frags_.size(); }

  size_t rowCount() const;

  const BufferFragment& getFragAt(const int frag_id) const {
    CHECK_LE(int(0), frag_id);
    CHECK_GT(buffer_frags_.size(), size_t(frag_id));
    return buffer_frags_[frag_id];
  }

  SQLTypeInfo getColType(const size_t col_idx) const {
    if (just_explain_) {
      return SQLTypeInfo(kTEXT, false);
    }
    return targets_[col_idx].sql_type;
  }

  bool definitelyHasNoRows() const { return buffer_frags_.empty() && !just_explain_ && !rowCount(); }

 private:
  void fuse(const IteratorTable& that);

  BufferFragment transformGroupByBuffer(const int64_t* group_by_buffer,
                                        const size_t groups_buffer_entry_count,
                                        const QueryMemoryDescriptor& query_mem_desc);

  const std::vector<TargetInfo> targets_;
  const QueryMemoryDescriptor query_mem_desc_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  std::vector<BufferFragment> buffer_frags_;
  const ExecutorDeviceType device_type_;

  const bool just_explain_;
  std::string explanation_;

  friend class QueryExecutionContext;
};

typedef std::unique_ptr<IteratorTable> IterTabPtr;

typedef boost::variant<RowSetPtr, IterTabPtr> ResultPtr;

enum RESPTR_TYPE {
  ROWSET,
  ITERTAB,
};

#endif  // QUERYENGINE_ITERATORTABLE_H
