#ifndef QUERYENGINE_GROUPBYANDAGGREGATE_H
#define QUERYENGINE_GROUPBYANDAGGREGATE_H

#include "BufferCompaction.h"
#include "CompilationOptions.h"
#include "GpuMemUtils.h"
#include "ResultRows.h"
#include "RuntimeFunctions.h"

#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"
#include "../Shared/checked_alloc.h"
#include "../Shared/sqltypes.h"

#include "SqlTypesLayout.h"

#include <boost/algorithm/string/join.hpp>
#include <glog/logging.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>

#include <vector>

inline const Analyzer::Expr* agg_arg(const Analyzer::Expr* expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(expr);
  return agg_expr ? agg_expr->get_arg() : nullptr;
}

inline bool constrained_not_null(const Analyzer::Expr* expr, const std::list<std::shared_ptr<Analyzer::Expr>>& quals) {
  for (const auto qual : quals) {
    auto uoper = std::dynamic_pointer_cast<Analyzer::UOper>(qual);
    if (!uoper) {
      continue;
    }
    bool is_negated{false};
    if (uoper->get_optype() == kNOT) {
      uoper = std::dynamic_pointer_cast<Analyzer::UOper>(uoper->get_own_operand());
      is_negated = true;
    }
    if (uoper && (uoper->get_optype() == kISNOTNULL || (is_negated && uoper->get_optype() == kISNULL))) {
      if (*uoper->get_own_operand() == *expr) {
        return true;
      }
    }
  }
  return false;
}

namespace {

inline void set_compact_type(TargetInfo& target, const SQLTypeInfo& new_type) {
  if (target.is_agg) {
    const auto agg_type = target.agg_kind;
    auto& agg_arg = target.agg_arg_type;
    if (agg_type != kCOUNT || agg_arg.get_type() != kNULLT) {
      agg_arg = new_type;
      return;
    }
  }
  target.sql_type = new_type;
}

inline std::pair<int64_t, int64_t> inline_int_max_min(const size_t byte_width) {
  switch (byte_width) {
    case 1:
      return std::make_pair(std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min());
    case 2:
      return std::make_pair(std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min());
    case 4:
      return std::make_pair(std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min());
    case 8:
      return std::make_pair(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min());
    default:
      CHECK(false);
  }
}

// TODO(alex): proper types for aggregate
inline int64_t get_agg_initial_val(const SQLAgg agg,
                                   const SQLTypeInfo& ti,
                                   const bool enable_compaction,
                                   const unsigned min_byte_width_to_compact) {
  CHECK(!ti.is_string());
  const auto byte_width = enable_compaction ? compact_byte_width(static_cast<unsigned>(get_bit_width(ti) >> 3),
                                                                 unsigned(min_byte_width_to_compact))
                                            : sizeof(int64_t);
  CHECK_GE(byte_width, static_cast<unsigned>(ti.get_size()));
  switch (agg) {
    case kAVG:
    case kSUM: {
      if (!ti.get_notnull()) {
        if (ti.is_fp()) {
          switch (byte_width) {
            case 4: {
              const float null_float = inline_fp_null_val(ti);
              return *reinterpret_cast<const int32_t*>(&null_float);
            }
            case 8: {
              const double null_double = inline_fp_null_val(ti);
              return *reinterpret_cast<const int64_t*>(&null_double);
            }
            default:
              CHECK(false);
          }
        } else {
          return inline_int_null_val(ti);
        }
      }
      switch (byte_width) {
        case 4: {
          const float zero_float{0.};
          return ti.is_fp() ? *reinterpret_cast<const int32_t*>(&zero_float) : 0;
        }
        case 8: {
          const double zero_double{0.};
          return ti.is_fp() ? *reinterpret_cast<const int64_t*>(&zero_double) : 0;
        }
        default:
          CHECK(false);
      }
    }
    case kCOUNT:
      return 0;
    case kMIN: {
      switch (byte_width) {
        case 4: {
          const float max_float = std::numeric_limits<float>::max();
          const float null_float = ti.is_fp() ? static_cast<float>(inline_fp_null_val(ti)) : 0.;
          return ti.is_fp() ? (ti.get_notnull() ? *reinterpret_cast<const int32_t*>(&max_float)
                                                : *reinterpret_cast<const int32_t*>(&null_float))
                            : (ti.get_notnull() ? std::numeric_limits<int32_t>::max() : inline_int_null_val(ti));
        }
        case 8: {
          const double max_double = std::numeric_limits<double>::max();
          const double null_double{ti.is_fp() ? inline_fp_null_val(ti) : 0.};
          return ti.is_fp() ? (ti.get_notnull() ? *reinterpret_cast<const int64_t*>(&max_double)
                                                : *reinterpret_cast<const int64_t*>(&null_double))
                            : (ti.get_notnull() ? std::numeric_limits<int64_t>::max() : inline_int_null_val(ti));
        }
        default:
          CHECK(false);
      }
    }
    case kMAX: {
      switch (byte_width) {
        case 4: {
          const float min_float = std::numeric_limits<float>::min();
          const float null_float = ti.is_fp() ? static_cast<float>(inline_fp_null_val(ti)) : 0.;
          return (ti.is_fp()) ? (ti.get_notnull() ? *reinterpret_cast<const int32_t*>(&min_float)
                                                  : *reinterpret_cast<const int32_t*>(&null_float))
                              : (ti.get_notnull() ? std::numeric_limits<int32_t>::min() : inline_int_null_val(ti));
        }
        case 8: {
          const double min_double = std::numeric_limits<double>::min();
          const double null_double{ti.is_fp() ? inline_fp_null_val(ti) : 0.};
          return ti.is_fp() ? (ti.get_notnull() ? *reinterpret_cast<const int64_t*>(&min_double)
                                                : *reinterpret_cast<const int64_t*>(&null_double))
                            : (ti.get_notnull() ? std::numeric_limits<int64_t>::min() : inline_int_null_val(ti));
        }
        default:
          CHECK(false);
      }
    }
    default:
      CHECK(false);
  }
}

inline std::vector<int64_t> init_agg_val_vec(const std::vector<TargetInfo>& targets,
                                             size_t agg_col_count,
                                             const bool is_group_by,
                                             const size_t min_byte_width_to_compact) {
  std::vector<int64_t> agg_init_vals(agg_col_count, 0);
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets.size() && agg_col_idx < agg_col_count;
       ++target_idx, ++agg_col_idx) {
    const auto agg_info = targets[target_idx];
    if (!agg_info.is_agg) {
      continue;
    }
    agg_init_vals[agg_col_idx] =
        get_agg_initial_val(agg_info.agg_kind, get_compact_type(agg_info), is_group_by, min_byte_width_to_compact);
    if (kAVG == agg_info.agg_kind) {
      agg_init_vals[++agg_col_idx] = 0;
    }
  }
  return agg_init_vals;
}

inline std::vector<int64_t> init_agg_val_vec(const std::vector<Analyzer::Expr*>& targets,
                                             const std::list<std::shared_ptr<Analyzer::Expr>>& quals,
                                             size_t agg_col_count,
                                             const bool is_group_by,
                                             const size_t min_byte_width_to_compact) {
  std::vector<TargetInfo> target_infos;
  target_infos.reserve(targets.size());
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets.size() && agg_col_idx < agg_col_count;
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = targets[target_idx];
    auto target = target_info(target_expr);
    auto arg_expr = agg_arg(target_expr);
    if (arg_expr && constrained_not_null(arg_expr, quals)) {
      target.skip_null_val = false;
      auto new_type = get_compact_type(target);
      new_type.set_notnull(true);
      set_compact_type(target, new_type);
    }
    target_infos.push_back(target);
  }
  return init_agg_val_vec(target_infos, agg_col_count, is_group_by, min_byte_width_to_compact);
}

inline int64_t get_initial_val(const TargetInfo& target_info, const size_t min_byte_width_to_compact) {
  if (!target_info.is_agg) {
    return 0;
  }
  const auto chosen_type = get_compact_type(target_info);
  return get_agg_initial_val(target_info.agg_kind, chosen_type, !chosen_type.is_fp(), min_byte_width_to_compact);
}

}  // namespace

class ReductionRanOutOfSlots : public std::runtime_error {
 public:
  ReductionRanOutOfSlots() : std::runtime_error("ReductionRanOutOfSlots") {}
};

class ColumnarResults {
 public:
  ColumnarResults(const ResultRows& rows, const size_t num_columns, const std::vector<SQLTypeInfo>& target_types)
      : column_buffers_(num_columns), num_rows_(rows.rowCount()), target_types_(target_types) {
    column_buffers_.resize(num_columns);
    for (size_t i = 0; i < num_columns; ++i) {
      CHECK(!target_types[i].is_array());
      CHECK(!target_types[i].is_string() ||
            (target_types[i].get_compression() == kENCODING_DICT && target_types[i].get_size() == 4));
      column_buffers_[i] = static_cast<const int8_t*>(checked_malloc(num_rows_ * (get_bit_width(target_types[i]) / 8)));
    }
    size_t row_idx{0};
    while (true) {
      const auto crt_row = rows.getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      for (size_t i = 0; i < num_columns; ++i) {
        const auto col_val = crt_row[i];
        const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
        CHECK(scalar_col_val);
        auto i64_p = boost::get<int64_t>(scalar_col_val);
        if (i64_p) {
          switch (get_bit_width(target_types[i])) {
            case 8:
              ((int8_t*)column_buffers_[i])[row_idx] = static_cast<int8_t>(*i64_p);
              break;
            case 16:
              ((int16_t*)column_buffers_[i])[row_idx] = static_cast<int16_t>(*i64_p);
              break;
            case 32:
              ((int32_t*)column_buffers_[i])[row_idx] = static_cast<int32_t>(*i64_p);
              break;
            case 64:
              ((int64_t*)column_buffers_[i])[row_idx] = *i64_p;
              break;
            default:
              CHECK(false);
          }
        } else {
          CHECK(target_types[i].is_fp());
          switch (target_types[i].get_type()) {
            case kFLOAT: {
              auto float_p = boost::get<float>(scalar_col_val);
              ((float*)column_buffers_[i])[row_idx] = static_cast<float>(*float_p);
              break;
            }
            case kDOUBLE: {
              auto double_p = boost::get<double>(scalar_col_val);
              ((double*)column_buffers_[i])[row_idx] = static_cast<double>(*double_p);
              break;
            }
            default:
              CHECK(false);
          }
        }
      }
      ++row_idx;
    }
  }

  ~ColumnarResults() {
    for (const auto column_buffer : column_buffers_) {
      free((void*)column_buffer);
    }
  }

  const std::vector<const int8_t*>& getColumnBuffers() const { return column_buffers_; }

  const size_t size() const { return num_rows_; }

  const SQLTypeInfo& getColumnType(const int col_id) const {
    CHECK_GE(col_id, 0);
    CHECK_LT(static_cast<size_t>(col_id), target_types_.size());
    return target_types_[col_id];
  }

 private:
  std::vector<const int8_t*> column_buffers_;
  const size_t num_rows_;
  const std::vector<SQLTypeInfo> target_types_;
};

namespace {

inline std::string nullable_str_to_string(const NullableString& str) {
  auto nptr = boost::get<void*>(&str);
  if (nptr) {
    CHECK(!*nptr);
    return "NULL";
  }
  auto sptr = boost::get<std::string>(&str);
  CHECK(sptr);
  return *sptr;
}

inline std::string datum_to_string(const TargetValue& tv, const SQLTypeInfo& ti, const std::string& delim) {
  if (ti.is_array()) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    CHECK(list_tv);
    std::vector<std::string> elem_strs;
    elem_strs.reserve(list_tv->size());
    const auto& elem_ti = ti.get_elem_type();
    for (const auto& elem_tv : *list_tv) {
      elem_strs.push_back(datum_to_string(elem_tv, elem_ti, delim));
    }
    return "{" + boost::algorithm::join(elem_strs, delim) + "}";
  }
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (ti.is_time()) {
    Datum datum;
    datum.timeval = *boost::get<int64_t>(scalar_tv);
    if (datum.timeval == NULL_BIGINT) {
      return "NULL";
    }
    return DatumToString(datum, ti);
  }
  if (ti.is_boolean()) {
    const auto bool_val = *boost::get<int64_t>(scalar_tv);
    return bool_val == NULL_BOOLEAN ? "NULL" : (bool_val ? "true" : "false");
  }
  auto iptr = boost::get<int64_t>(scalar_tv);
  if (iptr) {
    return *iptr == inline_int_null_val(ti) ? "NULL" : std::to_string(*iptr);
  }
  auto fptr = boost::get<float>(scalar_tv);
  if (fptr) {
    return *fptr == inline_fp_null_val(ti) ? "NULL" : std::to_string(*fptr);
  }
  auto dptr = boost::get<double>(scalar_tv);
  if (dptr) {
    return *dptr == inline_fp_null_val(ti.is_decimal() ? SQLTypeInfo(kDOUBLE, false) : ti) ? "NULL"
                                                                                           : std::to_string(*dptr);
  }
  auto sptr = boost::get<NullableString>(scalar_tv);
  CHECK(sptr);
  return nullable_str_to_string(*sptr);
}

class ScopedScratchBuffer {
 public:
  ScopedScratchBuffer(const size_t num_bytes, Data_Namespace::DataMgr* data_mgr, const int device_id)
      : data_mgr_(data_mgr), ab_(alloc_gpu_abstract_buffer(data_mgr_, num_bytes, device_id)) {}
  ~ScopedScratchBuffer() { data_mgr_->freeAllBuffers(); }
  CUdeviceptr getPtr() const { return reinterpret_cast<CUdeviceptr>(ab_->getMemoryPtr()); }

 private:
  Data_Namespace::DataMgr* data_mgr_;
  Data_Namespace::AbstractBuffer* ab_;
};

}  // namespace

inline std::string row_col_to_string(const ResultRows& rows,
                                     const size_t row_idx,
                                     const size_t i,
                                     const std::string& delim = ", ") {
  const auto tv = rows.getRowAt(row_idx, i, true);
  const auto ti = rows.getColType(i);
  return datum_to_string(tv, ti, delim);
}

inline std::string row_col_to_string(const std::vector<TargetValue>& row,
                                     const size_t i,
                                     const SQLTypeInfo& ti,
                                     const std::string& delim = ", ") {
  return datum_to_string(row[i], ti, delim);
}

class QueryExecutionContext : boost::noncopyable {
 public:
  // TODO(alex): move init_agg_vals to GroupByBufferDescriptor, remove device_type
  QueryExecutionContext(const QueryMemoryDescriptor&,
                        const std::vector<int64_t>& init_agg_vals,
                        const Executor* executor,
                        const ExecutorDeviceType device_type,
                        const int device_id,
                        const std::vector<std::vector<const int8_t*>>& col_buffers,
                        std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                        const bool output_columnar,
                        const bool sort_on_gpu,
                        RenderAllocatorMap*);

  // TOOD(alex): get rid of targets parameter
  ResultRows getRowSet(const std::vector<Analyzer::Expr*>& targets,
                       const QueryMemoryDescriptor& query_mem_desc,
                       const bool was_auto_device) const noexcept;
  ResultRows groupBufferToResults(const size_t i,
                                  const std::vector<Analyzer::Expr*>& targets,
                                  const bool was_auto_device) const;

  std::vector<int64_t*> launchGpuCode(const std::vector<void*>& cu_functions,
                                      const bool hoist_literals,
                                      const std::vector<int8_t>& literal_buff,
                                      std::vector<std::vector<const int8_t*>> col_buffers,
                                      const std::vector<int64_t>& num_rows,
                                      const std::vector<uint64_t>& frag_row_offsets,
                                      const int32_t scan_limit,
                                      const std::vector<int64_t>& init_agg_vals,
                                      Data_Namespace::DataMgr* data_mgr,
                                      const unsigned block_size_x,
                                      const unsigned grid_size_x,
                                      const int device_id,
                                      int32_t* error_code,
                                      const uint32_t num_tables,
                                      const int64_t join_hash_table,
                                      RenderAllocatorMap* render_allocator_map) const;

 private:
  bool isEmptyBin(const int64_t* group_by_buffer, const size_t bin, const size_t key_idx) const;
  void outputBin(ResultRows& results,
                 const std::vector<Analyzer::Expr*>& targets,
                 int64_t* group_by_buffer,
                 const size_t bin) const;

  void initColumnPerRow(int8_t* row_ptr,
                        const size_t bin,
                        const int64_t* init_vals,
                        const std::vector<ssize_t>& bitmap_sizes);

  void initGroups(int64_t* groups_buffer,
                  const int64_t* init_vals,
                  const int32_t groups_buffer_entry_count,
                  const bool keyless,
                  const size_t warp_size);

  template <typename T>
  int8_t* initColumnarBuffer(T* buffer_ptr,
                             const T init_val,
                             const uint32_t entry_count,
                             const ssize_t bitmap_sz = -1,
                             const bool key_or_col = true);

  void initColumnarGroups(int64_t* groups_buffer,
                          const int64_t* init_vals,
                          const int32_t groups_buffer_entry_count,
                          const bool keyless);
#ifdef HAVE_CUDA
  enum {
    COL_BUFFERS,
    NUM_FRAGMENTS,
    LITERALS,
    NUM_ROWS,
    FRAG_ROW_OFFSETS,
    MAX_MATCHED,
    TOTAL_MATCHED,
    INIT_AGG_VALS,
    GROUPBY_BUF,
    SMALL_BUF,
    ERROR_CODE,
    NUM_TABLES,
    JOIN_HASH_TABLE,
    KERN_PARAM_COUNT,
  };

  std::vector<CUdeviceptr> prepareKernelParams(const std::vector<std::vector<const int8_t*>>& col_buffers,
                                               const std::vector<int8_t>& literal_buff,
                                               const std::vector<int64_t>& num_rows,
                                               const std::vector<uint64_t>& frag_row_offsets,
                                               const int32_t scan_limit,
                                               const std::vector<int64_t>& init_agg_vals,
                                               const std::vector<int32_t>& error_codes,
                                               const uint32_t num_tables,
                                               const int64_t join_hash_table,
                                               Data_Namespace::DataMgr* data_mgr,
                                               const int device_id,
                                               const bool hoist_literals,
                                               const bool is_group_by) const;

  GpuQueryMemory prepareGroupByDevBuffer(Data_Namespace::DataMgr* data_mgr,
                                         RenderAllocator* render_allocator,
                                         const CUdeviceptr init_agg_vals_dev_ptr,
                                         const int device_id,
                                         const unsigned block_size_x,
                                         const unsigned grid_size_x,
                                         const bool can_sort_on_gpu) const;
#endif

  std::vector<ssize_t> allocateCountDistinctBuffers(const bool deferred);
  int64_t allocateCountDistinctBitmap(const size_t bitmap_sz);
  int64_t allocateCountDistinctSet();

  const QueryMemoryDescriptor& query_mem_desc_;
  std::vector<int64_t> init_agg_vals_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  const std::vector<std::vector<const int8_t*>>& col_buffers_;
  const size_t num_buffers_;

  std::vector<int64_t*> group_by_buffers_;
  std::vector<int64_t*> small_group_by_buffers_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const bool output_columnar_;
  const bool sort_on_gpu_;

  friend class Executor;
  friend void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                             const QueryExecutionContext* query_exe_context,
                                             const GpuQueryMemory& gpu_query_mem,
                                             const unsigned block_size_x,
                                             const unsigned grid_size_x,
                                             const int device_id,
                                             const bool prepend_index_buffer);
};

class GroupByAndAggregate {
 public:
  GroupByAndAggregate(Executor* executor,
                      const ExecutorDeviceType device_type,
                      const RelAlgExecutionUnit& ra_exe_unit,
                      const bool render_output,
                      const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                      std::shared_ptr<RowSetMemoryOwner>,
                      const size_t max_groups_buffer_entry_count,
                      const size_t small_groups_buffer_entry_count,
                      const int8_t crt_min_byte_width,
                      const bool allow_multifrag,
                      const bool output_columnar_hint);

  const QueryMemoryDescriptor& getQueryMemoryDescriptor() const;

  bool outputColumnar() const;

  void patchGroupbyCall(llvm::CallInst* call_site);

  // returns true iff checking the error code after every row
  // is required -- slow path group by queries for now
  bool codegen(llvm::Value* filter_result, const CompilationOptions& co);

 private:
  struct ColRangeInfo {
    const GroupByColRangeType hash_type_;
    const int64_t min;
    const int64_t max;
    const int64_t bucket;
    const bool has_nulls;
  };

  struct DiamondCodegen {
    DiamondCodegen(llvm::Value* cond,
                   Executor* executor,
                   const bool chain_to_next,
                   const std::string& label_prefix,
                   DiamondCodegen* parent = nullptr);
    void setChainToNext();
    void setFalseTarget(llvm::BasicBlock* cond_false);
    ~DiamondCodegen();

    Executor* executor_;
    llvm::BasicBlock* cond_true_;
    llvm::BasicBlock* cond_false_;
    llvm::BasicBlock* orig_cond_false_;
    bool chain_to_next_;
    DiamondCodegen* parent_;
  };

  bool gpuCanHandleOrderEntries(const std::list<Analyzer::OrderEntry>& order_entries);

  void initQueryMemoryDescriptor(const bool allow_multifrag,
                                 const size_t max_groups_buffer_entry_count,
                                 const size_t small_groups_buffer_entry_count,
                                 const int8_t crt_min_byte_width,
                                 const bool sort_on_gpu_hint,
                                 const bool render_output);
  void addTransientStringLiterals();

  CountDistinctDescriptors initCountDistinctDescriptors();

  std::tuple<llvm::Value*, llvm::Value*> codegenGroupBy(const CompilationOptions&, DiamondCodegen&);

  llvm::Function* codegenPerfectHashFunction();

  GroupByAndAggregate::ColRangeInfo getColRangeInfo();

  GroupByAndAggregate::ColRangeInfo getExprRangeInfo(const Analyzer::Expr* expr) const;

  static int64_t getBucketedCardinality(const GroupByAndAggregate::ColRangeInfo& col_range_info);

  struct KeylessInfo {
    const bool keyless;
    const int32_t target_index;
    const int64_t init_val;
  };

  KeylessInfo getKeylessInfo(const std::vector<Analyzer::Expr*>& target_expr_list, const bool is_group_by) const;

  llvm::Value* convertNullIfAny(const SQLTypeInfo& arg_type,
                                const SQLTypeInfo& agg_type,
                                const size_t chosen_bytes,
                                llvm::Value* target);
#ifdef ENABLE_COMPACTION
  bool detectOverflowAndUnderflow(llvm::Value* agg_addr,
                                  llvm::Value* val,
                                  const TargetInfo& agg_info,
                                  const size_t chosen_bytes,
                                  const bool need_skip_null,
                                  const std::string& agg_base_name);
#endif
  bool codegenAggCalls(const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                       const std::vector<llvm::Value*>& agg_out_vec,
                       const CompilationOptions&);

  void codegenCountDistinct(const size_t target_idx,
                            const Analyzer::Expr* target_expr,
                            std::vector<llvm::Value*>& agg_args,
                            const QueryMemoryDescriptor&,
                            const ExecutorDeviceType);

  std::vector<llvm::Value*> codegenAggArg(const Analyzer::Expr* target_expr, const CompilationOptions& co);

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  QueryMemoryDescriptor query_mem_desc_;
  Executor* executor_;
  const RelAlgExecutionUnit& ra_exe_unit_;
  const std::vector<Fragmenter_Namespace::TableInfo>& query_infos_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  bool output_columnar_;

  friend class Executor;
};

namespace {

inline std::vector<Analyzer::Expr*> get_agg_target_exprs(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& target_list) {
  std::vector<Analyzer::Expr*> result;
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    result.push_back(target_expr);
  }
  return result;
}

inline std::vector<Analyzer::Expr*> get_agg_target_exprs(const Planner::Plan* plan) {
  const auto& target_list = plan->get_targetlist();
  return get_agg_target_exprs(target_list);
}

inline int64_t extract_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (type) {
    case kBOOLEAN:
      return datum.tinyintval;
    case kSMALLINT:
      return datum.smallintval;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
    case kINT:
      return datum.intval;
    case kBIGINT:
      return datum.bigintval;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return datum.timeval;
    default:
      CHECK(false);
  }
}

inline int64_t extract_min_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.min, ti);
}

inline int64_t extract_max_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.max, ti);
}

}  // namespace

template <class T>
inline std::vector<int8_t> get_col_byte_widths(const T& col_expr_list) {
  std::vector<int8_t> col_widths;
  for (const auto col_expr : col_expr_list) {
    if (!col_expr) {
      // row index
      col_widths.push_back(sizeof(int64_t));
    } else {
      const auto agg_info = target_info(col_expr);
      const auto chosen_type = get_compact_type(agg_info);
      if ((chosen_type.is_string() && chosen_type.get_compression() == kENCODING_NONE) || chosen_type.is_array()) {
        col_widths.push_back(sizeof(int64_t));
        col_widths.push_back(sizeof(int64_t));
        continue;
      }
      const auto col_expr_bitwidth = get_bit_width(chosen_type);
      CHECK_EQ(size_t(0), col_expr_bitwidth % 8);
      col_widths.push_back(static_cast<int8_t>(col_expr_bitwidth >> 3));
      // for average, we'll need to keep the count as well
      if (agg_info.agg_kind == kAVG) {
        CHECK(agg_info.is_agg);
        col_widths.push_back(sizeof(int64_t));
      }
    }
  }
  return col_widths;
}

inline int8_t get_min_byte_width() {
#ifdef ENABLE_COMPACTION
  return 4;
#else
  return MAX_BYTE_WIDTH_SUPPORTED;
#endif
}

struct RelAlgExecutionUnit;

int8_t pick_target_compact_width(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                 const int8_t crt_min_byte_width);

#endif  // QUERYENGINE_GROUPBYANDAGGREGATE_H
