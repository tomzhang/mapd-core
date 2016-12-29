#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include "BufferCompaction.h"
#include "GroupByAndAggregate.h"
#include "InValuesBitmap.h"
#include "InputMetadata.h"
#include "JoinHashTable.h"
#include "LLVMGlobalContext.h"
#include "NvidiaKernel.h"
#include "RelAlgExecutionUnit.h"
#include "TargetMetaInfo.h"

#include "../Analyzer/Analyzer.h"
#include "../Chunk/Chunk.h"
#include "../Planner/Planner.h"
#include "../Shared/measure.h"
#include "../Shared/thread_count.h"
#include "../StringDictionary/StringDictionary.h"

#include <boost/functional/hash.hpp>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <rapidjson/document.h>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <stack>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

extern bool g_enable_watchdog;

struct RenderInfo {
  bool do_render;
  const int session_id;
  const int render_widget_id;
  std::unique_ptr<RenderAllocatorMap> render_allocator_map_ptr;
  const std::string render_vega;

  std::shared_ptr<QueryRenderer::QueryDataLayout> result_query_data_layout;

  RenderInfo(const int session_id,
             const int render_widget_id,
             const std::string& render_vega = "",
             const bool do_render = true)
      : do_render(do_render), session_id(session_id), render_widget_id(render_widget_id), render_vega(render_vega) {}
};

class WatchdogException : public std::runtime_error {
 public:
  WatchdogException(const std::string& cause) : std::runtime_error(cause) {}
};

class Executor;

inline llvm::Type* get_int_type(const int width, llvm::LLVMContext& context) {
  switch (width) {
    case 64:
      return llvm::Type::getInt64Ty(context);
    case 32:
      return llvm::Type::getInt32Ty(context);
      break;
    case 16:
      return llvm::Type::getInt16Ty(context);
      break;
    case 8:
      return llvm::Type::getInt8Ty(context);
      break;
    case 1:
      return llvm::Type::getInt1Ty(context);
      break;
    default:
      LOG(FATAL) << "Unsupported integer width: " << width;
  }
}

inline llvm::Value* get_arg_by_name(llvm::Function* func, const std::string& name) {
  auto& arg_list = func->getArgumentList();
  for (auto& arg : arg_list) {
    if (arg.getName() == name) {
      return &arg;
    }
  }
  CHECK(false);
  return nullptr;
}

inline uint32_t log2_bytes(const uint32_t bytes) {
  switch (bytes) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 4:
      return 2;
    case 8:
      return 3;
    default:
      abort();
  }
}

inline const ColumnDescriptor* get_column_descriptor(const int col_id,
                                                     const int table_id,
                                                     const Catalog_Namespace::Catalog& cat) {
  CHECK_GT(table_id, 0);
  const auto col_desc = cat.getMetadataForColumn(table_id, col_id);
  CHECK(col_desc);
  return col_desc;
}

inline const Analyzer::Expr* extract_cast_arg(const Analyzer::Expr* expr) {
  const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(expr);
  if (!cast_expr || cast_expr->get_optype() != kCAST) {
    return expr;
  }
  return cast_expr->get_operand();
}

inline std::string numeric_type_name(const SQLTypeInfo& ti) {
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_boolean() || ti.is_time() || ti.is_fp() ||
        (ti.is_string() && ti.get_compression() == kENCODING_DICT) || ti.is_timeinterval());
  if (ti.is_integer() || ti.is_decimal() || ti.is_boolean() || ti.is_time() || ti.is_string() || ti.is_timeinterval()) {
    return "int" + std::to_string(ti.get_logical_size() * 8) + "_t";
  }
  return ti.get_type() == kDOUBLE ? "double" : "float";
}

inline const ColumnDescriptor* get_column_descriptor_maybe(const int col_id,
                                                           const int table_id,
                                                           const Catalog_Namespace::Catalog& cat) {
  CHECK(table_id);
  return table_id > 0 ? get_column_descriptor(col_id, table_id, cat) : nullptr;
}

inline const ResultPtr& get_temporary_table(const TemporaryTables* temporary_tables, const int table_id) {
  CHECK_LT(table_id, 0);
  const auto it = temporary_tables->find(table_id);
  CHECK(it != temporary_tables->end());
  const auto& temp = it->second;
  CHECK(boost::get<RowSetPtr>(&temp) || boost::get<IterTabPtr>(&temp));
  return temp;
}

inline const SQLTypeInfo get_column_type(const int col_id,
                                         const int table_id,
                                         const ColumnDescriptor* cd,
                                         const TemporaryTables* temporary_tables) {
  CHECK(cd || temporary_tables);
  if (cd) {
    CHECK_EQ(col_id, cd->columnId);
    CHECK_EQ(table_id, cd->tableId);
    return cd->columnType;
  }
  const auto& temp = get_temporary_table(temporary_tables, table_id);
  if (const auto rows = boost::get<RowSetPtr>(&temp)) {
    CHECK(*rows);
    return (*rows)->getColType(col_id);
  } else if (const auto tab = boost::get<IterTabPtr>(&temp)) {
    CHECK(*tab);
    return (*tab)->getColType(col_id);
  }

  abort();
}

template <typename PtrTy>
inline const ColumnarResults* rows_to_columnar_results(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                                       const PtrTy& result,
                                                       const int number) {
  std::vector<SQLTypeInfo> col_types;
  for (size_t i = 0; i < result->colCount(); ++i) {
    col_types.push_back(result->getColType(i));
  }
  return new ColumnarResults(row_set_mem_owner, *result, number, col_types);
}

// TODO(alex): Adjust interfaces downstream and make this not needed.
inline std::vector<Analyzer::Expr*> get_exprs_not_owned(const std::vector<std::shared_ptr<Analyzer::Expr>>& exprs) {
  std::vector<Analyzer::Expr*> exprs_not_owned;
  for (const auto expr : exprs) {
    exprs_not_owned.push_back(expr.get());
  }
  return exprs_not_owned;
}

inline const ColumnarResults* columnarize_result(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                                 const ResultPtr& result,
                                                 const int frag_id) {
  if (const auto rows = boost::get<RowSetPtr>(&result)) {
    CHECK_EQ(0, frag_id);
    return rows_to_columnar_results(row_set_mem_owner, *rows, (*rows)->colCount());
  } else if (const auto table = boost::get<IterTabPtr>(&result)) {
    return rows_to_columnar_results(row_set_mem_owner, *table, frag_id);
  }
  CHECK(false);
  return nullptr;
}

class CompilationRetryNoLazyFetch : public std::runtime_error {
 public:
  CompilationRetryNoLazyFetch() : std::runtime_error("CompilationRetryNoLazyFetch") {}
};

class TooManyLiterals : public std::runtime_error {
 public:
  TooManyLiterals() : std::runtime_error("Too many literals in the query") {}
};

class CompilationRetryNoCompaction : public std::runtime_error {
 public:
  CompilationRetryNoCompaction() : std::runtime_error("CompilationRetryNoCompaction") {}
};

class ExtensionFunction;

namespace std {
template <>
struct hash<std::vector<int>> {
  size_t operator()(const std::vector<int>& vec) const {
    return vec.size() ^ boost::hash_range(vec.begin(), vec.end());
  }
};

template <>
struct hash<std::pair<int, int>> {
  size_t operator()(const std::pair<int, int>& p) const { return boost::hash<std::pair<int, int>>()(p); }
};

}  // std

class Executor {
  static_assert(sizeof(float) == 4 && sizeof(double) == 8,
                "Host hardware not supported, unexpected size of float / double.");

 public:
  Executor(const int db_id,
           const size_t block_size_x,
           const size_t grid_size_x,
           const std::string& debug_dir,
           const std::string& debug_file,
           ::QueryRenderer::QueryRenderManager* render_manager);

  static std::shared_ptr<Executor> getExecutor(const int db_id,
                                               const std::string& debug_dir = "",
                                               const std::string& debug_file = "",
                                               const size_t block_size_x = 0,
                                               const size_t grid_size_x = 0,
                                               ::QueryRenderer::QueryRenderManager* render_manager = nullptr);

  static void nukeCacheOfExecutors() {
    std::lock_guard<std::mutex> flush_lock(execute_mutex_);  // don't want native code to vanish while executing
    mapd_unique_lock<mapd_shared_mutex> lock(executors_cache_mutex_);
    (decltype(executors_){}).swap(executors_);
  }

  typedef std::tuple<std::string, const Analyzer::Expr*, int64_t, const size_t> AggInfo;

  ResultRows execute(const Planner::RootPlan* root_plan,
                     const Catalog_Namespace::SessionInfo& session,
                     const bool hoist_literals,
                     const ExecutorDeviceType device_type,
                     const ExecutorOptLevel,
                     const bool allow_multifrag,
                     const bool allow_loop_joins,
                     RenderInfo* render_query_data = nullptr);

  int64_t getRowidForPixel(const int64_t x,
                           const int64_t y,
                           const int session_id,
                           const int render_widget_id,
                           const int pixelRadius = 0);

  ResultRows renderPolygons(const std::string& queryStr,
                            const ResultRows& rows,
                            const std::vector<TargetMetaInfo>& row_shape,
                            const Catalog_Namespace::SessionInfo& session,
                            const int render_widget_id,
                            const rapidjson::Value& data_desc,
                            const std::string* render_config_json = nullptr,
                            const bool is_projection_query = true,
                            const std::string& poly_table_name = "");

#ifdef HAVE_RENDERING
  int32_t getStringId(const std::string& table_name,
                      const std::string& col_name,
                      const std::string& col_val,
                      const ::QueryRenderer::QueryDataLayout*) const;

  std::vector<double> getShapeVertices(const Catalog_Namespace::SessionInfo& session,
                                       const TableDescriptor* td,
                                       const std::string& shape_col_group,
                                       QueryRenderer::PolyRowDataShPtr& rowDataPtr);

  std::vector<unsigned> getShapeIndices(const Catalog_Namespace::SessionInfo& session,
                                        const TableDescriptor* td,
                                        const std::string& shape_col_group,
                                        QueryRenderer::PolyRowDataShPtr& rowDataPtr);

  struct LineDrawData {
    std::vector<::Rendering::GL::Resources::IndirectDrawVertexData> data;
    const std::vector<size_t> offsets;
  };

  LineDrawData getShapeLineDrawData(const Catalog_Namespace::SessionInfo& session,
                                    const TableDescriptor* td,
                                    const std::string& shape_col_group,
                                    QueryRenderer::PolyRowDataShPtr& rowDataPtr);

  std::vector<::Rendering::GL::Resources::IndirectDrawIndexData> getShapePolyDrawData(
      const Catalog_Namespace::SessionInfo& session,
      const TableDescriptor* td,
      const std::string& shape_col_group,
      QueryRenderer::PolyRowDataShPtr& rowDataPtr);

  struct PolyRenderDataQueryResult {
    std::shared_ptr<::QueryRenderer::QueryDataLayout> poly_render_data_layout;
    std::unique_ptr<char[]> data;
    const size_t num_data_bytes;
    const size_t align_bytes;
  };

  PolyRenderDataQueryResult getPolyRenderDataTemplate(const std::vector<TargetMetaInfo>& row_shape,
                                                      const size_t entry_count,
                                                      const size_t gpuId,
                                                      const size_t rowid_idx,
                                                      const bool is_projection_query);

  void setPolyRenderDataEntry(PolyRenderDataQueryResult& render_data,
                              const std::vector<TargetValue>& row,
                              const std::vector<TargetMetaInfo>& row_shape,
                              const size_t polyidx,
                              const size_t rowidx,
                              const size_t rowid_idx,
                              const size_t align_bytes,
                              const bool is_projection_query);

#endif  // HAVE_RENDERING

  StringDictionary* getStringDictionary(const int dictId,
                                        const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const;

  bool isCPUOnly() const;

  bool isArchMaxwell(const ExecutorDeviceType dt) const;

  bool isOuterJoin() const { return cgen_state_->is_outer_join_; }

  const ColumnDescriptor* getColumnDescriptor(const Analyzer::ColumnVar*) const;

  const Catalog_Namespace::Catalog* getCatalog() const;

  const TemporaryTables* getTemporaryTables() const;

  Fragmenter_Namespace::TableInfo getTableInfo(const int table_id);

  typedef boost::variant<int8_t, int16_t, int32_t, int64_t, float, double, std::pair<std::string, int>, std::string>
      LiteralValue;
  typedef std::vector<LiteralValue> LiteralValues;

 private:
  void clearInputTableInfoCache();

  template <class T>
  llvm::ConstantInt* ll_int(const T v) const {
    return static_cast<llvm::ConstantInt*>(
        llvm::ConstantInt::get(get_int_type(sizeof(v) * 8, cgen_state_->context_), v));
  }
  llvm::ConstantFP* ll_fp(const float v) const {
    return static_cast<llvm::ConstantFP*>(llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_), v));
  }
  llvm::ConstantFP* ll_fp(const double v) const {
    return static_cast<llvm::ConstantFP*>(llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_), v));
  }
  llvm::ConstantInt* ll_bool(const bool v) const {
    return static_cast<llvm::ConstantInt*>(llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), v));
  }
  std::vector<llvm::Value*> codegen(const Analyzer::Expr*, const bool fetch_columns, const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::BinOper*, const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::UOper*, const CompilationOptions&);
  std::vector<llvm::Value*> codegen(const Analyzer::ColumnVar*, const bool fetch_column, const bool hoist_literals);
  std::vector<llvm::Value*> codegenColVar(const Analyzer::ColumnVar*,
                                          const bool fetch_column,
                                          const bool hoist_literals);
  llvm::Value* codgenAdjustFixedEncNull(llvm::Value*, const SQLTypeInfo&);
  std::vector<llvm::Value*> codegenOuterJoinNullPlaceholder(const std::vector<llvm::Value*>&,
                                                            const Analyzer::ColumnVar*);
  llvm::Value* resolveGroupedColumnReference(const Analyzer::ColumnVar*);
  std::vector<llvm::Value*> codegen(const Analyzer::Constant*,
                                    const EncodingType enc_type,
                                    const int dict_id,
                                    const CompilationOptions&);
  std::vector<llvm::Value*> codegenHoistedConstants(const std::vector<const Analyzer::Constant*>&,
                                                    const EncodingType enc_type,
                                                    const int dict_id);
  int deviceCount(const ExecutorDeviceType) const;
  std::vector<llvm::Value*> codegen(const Analyzer::CaseExpr*, const CompilationOptions&);
  llvm::Value* codegenCase(const Analyzer::CaseExpr*,
                           llvm::Type* case_llvm_type,
                           const bool is_real_str,
                           const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::ExtractExpr*, const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::DatediffExpr*, const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::DatetruncExpr*, const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::CharLengthExpr*, const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::LikeExpr*, const CompilationOptions&);
  llvm::Value* codegenDictLike(const std::shared_ptr<Analyzer::Expr> arg,
                               const Analyzer::Constant* pattern,
                               const bool ilike,
                               const bool is_simple,
                               const char escape_char,
                               const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::RegexpExpr*, const CompilationOptions&);
  llvm::Value* codegenDictRegexp(const std::shared_ptr<Analyzer::Expr> arg,
                                 const Analyzer::Constant* pattern,
                                 const char escape_char,
                                 const CompilationOptions&);
  llvm::Value* codegen(const Analyzer::InValues*, const CompilationOptions&);
  std::unique_ptr<InValuesBitmap> createInValuesBitmap(const Analyzer::InValues*, const CompilationOptions&);
  llvm::Value* codegenCmp(const Analyzer::BinOper*, const CompilationOptions&);
  llvm::Value* codegenCmp(const SQLOps,
                          const SQLQualifier,
                          std::vector<llvm::Value*>,
                          const SQLTypeInfo&,
                          const Analyzer::Expr*,
                          const CompilationOptions&);
  llvm::Value* codegenQualifierCmp(const SQLOps,
                                   const SQLQualifier,
                                   std::vector<llvm::Value*>,
                                   const Analyzer::Expr*,
                                   const CompilationOptions&);
  llvm::Value* codegenLogicalShortCircuit(const Analyzer::BinOper*, const CompilationOptions&);
  llvm::Value* codegenLogical(const Analyzer::BinOper*, const CompilationOptions&);
  llvm::Value* toBool(llvm::Value*);
  llvm::Value* codegenArith(const Analyzer::BinOper*, const CompilationOptions&);
  bool checkExpressionRanges(const Analyzer::UOper*, int64_t, int64_t);
  bool checkExpressionRanges(const Analyzer::BinOper*, int64_t, int64_t);
  llvm::Value* codegenAdd(const Analyzer::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&);
  llvm::Value* codegenSub(const Analyzer::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&);
  llvm::Value* codegenMul(const Analyzer::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&,
                          bool downscale = true);
  llvm::Value* codegenDeciMul(const Analyzer::BinOper*, bool, const CompilationOptions&);
  llvm::Value* codegenDiv(llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&);
  llvm::Value* codegenMod(llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&);
  llvm::Value* codegenLogical(const Analyzer::UOper*, const CompilationOptions&);
  llvm::Value* codegenCast(const Analyzer::UOper*, const CompilationOptions&);
  llvm::Value* codegenCast(llvm::Value* operand_lv,
                           const SQLTypeInfo& operand_ti,
                           const SQLTypeInfo& ti,
                           const bool operand_is_const);
  llvm::Value* codegenCastTimestampToDate(llvm::Value* ts_lv, const bool nullable);
  llvm::Value* codegenCastFromString(llvm::Value* operand_lv,
                                     const SQLTypeInfo& operand_ti,
                                     const SQLTypeInfo& ti,
                                     const bool operand_is_const);
  llvm::Value* codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                          const SQLTypeInfo& operand_ti,
                                          const SQLTypeInfo& ti,
                                          bool upscale = true);
  llvm::Value* codegenCastToFp(llvm::Value* operand_lv, const SQLTypeInfo& operand_ti, const SQLTypeInfo& ti);
  llvm::Value* codegenCastFromFp(llvm::Value* operand_lv, const SQLTypeInfo& operand_ti, const SQLTypeInfo& ti);
  llvm::Value* codegenUMinus(const Analyzer::UOper*, const CompilationOptions&);
  llvm::Value* codegenIsNull(const Analyzer::UOper*, const CompilationOptions&);
  llvm::Value* codegenIsNullNumber(llvm::Value*, const SQLTypeInfo&);
  llvm::Value* codegenUnnest(const Analyzer::UOper*, const CompilationOptions&);
  llvm::Value* codegenArrayAt(const Analyzer::BinOper*, const CompilationOptions&);

  llvm::Value* codegenFunctionOper(const Analyzer::FunctionOper*, const CompilationOptions&);

  struct ArgNullcheckBBs {
    llvm::BasicBlock* args_null_bb;
    llvm::BasicBlock* args_notnull_bb;
    llvm::BasicBlock* orig_bb;
  };

  ArgNullcheckBBs beginArgsNullcheck(const Analyzer::FunctionOper* function_oper,
                                     const std::vector<llvm::Value*>& orig_arg_lvs);

  llvm::Value* endArgsNullcheck(const ArgNullcheckBBs&, llvm::Value*, const Analyzer::FunctionOper*);

  llvm::Value* codegenFunctionOperWithCustomTypeHandling(const Analyzer::FunctionOperWithCustomTypeHandling*,
                                                         const CompilationOptions&);
  llvm::Value* codegenFunctionOperNullArg(const Analyzer::FunctionOper*, const std::vector<llvm::Value*>&);
  std::vector<llvm::Value*> codegenFunctionOperCastArgs(const Analyzer::FunctionOper*,
                                                        const ExtensionFunction*,
                                                        const std::vector<llvm::Value*>&);
  llvm::ConstantInt* codegenIntConst(const Analyzer::Constant* constant);
  llvm::Value* colByteStream(const Analyzer::ColumnVar* col_var, const bool fetch_column, const bool hoist_literals);
  llvm::Value* posArg(const Analyzer::Expr*) const;
  const Analyzer::ColumnVar* hashJoinLhs(const Analyzer::ColumnVar* rhs) const;
  llvm::Value* fragRowOff() const;
  llvm::Value* rowsPerScan() const;
  llvm::ConstantInt* inlineIntNull(const SQLTypeInfo&);
  llvm::ConstantFP* inlineFpNull(const SQLTypeInfo&);
  std::pair<llvm::ConstantInt*, llvm::ConstantInt*> inlineIntMaxMin(const size_t byte_width, const bool is_signed);

  RowSetPtr executeSelectPlan(const Planner::Plan* plan,
                              const int64_t limit,
                              const int64_t offset,
                              const bool hoist_literals,
                              const ExecutorDeviceType device_type,
                              const ExecutorOptLevel,
                              const Catalog_Namespace::Catalog&,
                              size_t& max_groups_buffer_entry_guess,
                              int32_t* error_code,
                              const Planner::Sort* sort_plan,
                              const bool allow_multifrag,
                              const bool just_explain,
                              const bool allow_loop_joins,
                              RenderAllocatorMap* render_allocator_map);

  struct CompilationResult {
    std::vector<void*> native_functions;
    std::unordered_map<int, LiteralValues> literal_values;
    QueryMemoryDescriptor query_mem_desc;
    bool output_columnar;
    std::string llvm_ir;
  };

  bool isArchPascal(const ExecutorDeviceType dt) const {
    return dt == ExecutorDeviceType::GPU && catalog_->get_dataMgr().cudaMgr_->isArchPascal();
  }

  enum class JoinImplType { Invalid, Loop, HashOneToOne };

  struct JoinInfo {
    JoinInfo(const JoinImplType join_impl_type,
             const std::vector<std::shared_ptr<Analyzer::BinOper>>& equi_join_tautologies,
             std::shared_ptr<JoinHashTable> join_hash_table,
             const std::string& hash_join_fail_reason)
        : join_impl_type_(join_impl_type),
          equi_join_tautologies_(equi_join_tautologies),
          join_hash_table_(join_hash_table),
          hash_join_fail_reason_(hash_join_fail_reason) {}

    JoinImplType join_impl_type_;
    std::vector<std::shared_ptr<Analyzer::BinOper>> equi_join_tautologies_;  // expressions we equi-join on are true by
                                                                             // definition when using a hash join; we'll
                                                                             // fold them to true during code generation
    std::shared_ptr<JoinHashTable> join_hash_table_;
    std::string hash_join_fail_reason_;
  };

  typedef std::deque<Fragmenter_Namespace::FragmentInfo> TableFragments;

  class ExecutionDispatch {
   private:
    Executor* executor_;
    const RelAlgExecutionUnit& ra_exe_unit_;
    const std::vector<InputTableInfo>& query_infos_;
    const Catalog_Namespace::Catalog& cat_;
    CompilationOptions co_;
    CompilationResult compilation_result_cpu_;
    CompilationResult compilation_result_gpu_;
    std::vector<uint64_t> all_frag_row_offsets_;
    std::vector<std::unique_ptr<QueryExecutionContext>> query_contexts_;
    std::vector<std::mutex> query_context_mutexes_;
    const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
    int32_t* error_code_;
    RenderAllocatorMap* render_allocator_map_;
    std::vector<std::pair<ResultPtr, std::vector<size_t>>> all_fragment_results_;
    static std::mutex reduce_mutex_;

    typedef std::vector<int> CacheKey;
    mutable std::mutex columnar_conversion_mutex_;
    mutable std::unordered_map<int, std::unordered_map<int, std::unique_ptr<const ColumnarResults>>>
        columnarized_table_cache_;
    mutable std::unordered_map<InputColDescriptor, std::unordered_map<CacheKey, std::unique_ptr<const ColumnarResults>>>
        columnarized_ref_table_cache_;

    const int8_t* getColumn(const ResultPtr& buffer,
                            const int table_id,
                            const int frag_id,
                            const int col_id,
                            const Data_Namespace::MemoryLevel memory_level,
                            const int device_id) const;
    uint64_t getFragOffset(const int frag_id, const int table_id) const;

    void runImpl(const ExecutorDeviceType chosen_device_type,
                 int chosen_device_id,
                 const std::map<int, std::vector<size_t>>& frag_ids,
                 const size_t ctx_idx,
                 const int64_t rowid_lookup_key);

   public:
    ExecutionDispatch(Executor* executor,
                      const RelAlgExecutionUnit& ra_exe_unit,
                      const std::vector<InputTableInfo>& query_infos,
                      const Catalog_Namespace::Catalog& cat,
                      const CompilationOptions& co,
                      const size_t context_count,
                      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                      int32_t* error_code,
                      RenderAllocatorMap* render_allocator_map);

    ExecutionDispatch(const ExecutionDispatch&) = delete;

    ExecutionDispatch& operator=(const ExecutionDispatch&) = delete;

    ExecutionDispatch(ExecutionDispatch&&) = delete;

    ExecutionDispatch& operator=(ExecutionDispatch&&) = delete;

    int8_t compile(const JoinInfo& join_info,
                   const size_t max_groups_buffer_entry_guess,
                   const int8_t crt_min_byte_width,
                   const ExecutionOptions& options,
                   const bool has_cardinality_estimation);

    void run(const ExecutorDeviceType chosen_device_type,
             int chosen_device_id,
             const std::map<int, std::vector<size_t>>& frag_ids,
             const size_t ctx_idx,
             const int64_t rowid_lookup_key) noexcept;

    const int8_t* getScanColumn(const int table_id,
                                const int frag_id,
                                const int col_id,
                                const std::map<int, const TableFragments*>& all_tables_fragments,
                                std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
                                std::list<ChunkIter>& chunk_iter_holder,
                                const Data_Namespace::MemoryLevel memory_level,
                                const int device_id) const;

    const int8_t* getColumn(const InputColDescriptor* col_desc,
                            const int frag_id,
                            const std::map<int, const TableFragments*>& all_tables_fragments,
                            const Data_Namespace::MemoryLevel memory_level,
                            const int device_id,
                            const bool is_rowid) const;

    static const int8_t* getColumn(const ColumnarResults* columnar_results,
                                   const int col_id,
                                   Data_Namespace::DataMgr* data_mgr,
                                   const Data_Namespace::MemoryLevel memory_level,
                                   const int device_id);

    std::string getIR(const ExecutorDeviceType device_type) const;

    ExecutorDeviceType getDeviceType() const;

    const RelAlgExecutionUnit& getExecutionUnit() const;

    const QueryMemoryDescriptor& getQueryMemoryDescriptor() const;

    const bool outputColumnar() const;

    const std::vector<uint64_t>& getFragOffsets() const;

    const std::vector<std::unique_ptr<QueryExecutionContext>>& getQueryContexts() const;

    std::vector<std::pair<ResultPtr, std::vector<size_t>>>& getFragmentResults();
  };

  ResultPtr executeWorkUnit(int32_t* error_code,
                            size_t& max_groups_buffer_entry_guess,
                            const bool is_agg,
                            const std::vector<InputTableInfo>&,
                            const RelAlgExecutionUnit&,
                            const CompilationOptions&,
                            const ExecutionOptions& options,
                            const Catalog_Namespace::Catalog&,
                            std::shared_ptr<RowSetMemoryOwner>,
                            RenderAllocatorMap* render_allocator_map,
                            const bool has_cardinality_estimation);

  RowSetPtr executeExplain(const ExecutionDispatch&);

  // TODO(alex): remove
  ExecutorDeviceType getDeviceTypeForTargets(const RelAlgExecutionUnit& ra_exe_unit,
                                             const ExecutorDeviceType requested_device_type);

  RowSetPtr collectAllDeviceResults(ExecutionDispatch& execution_dispatch,
                                    const std::vector<Analyzer::Expr*>& target_exprs,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                    const bool output_columnar);

  std::string renderRows(const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets,
                         RenderInfo* render_query_data);

  void dispatchFragments(const std::function<void(const ExecutorDeviceType chosen_device_type,
                                                  int chosen_device_id,
                                                  const std::map<int, std::vector<size_t>>& frag_ids,
                                                  const size_t ctx_idx,
                                                  const int64_t rowid_lookup_key)> dispatch,
                         const ExecutionDispatch& execution_dispatch,
                         const ExecutionOptions& eo,
                         const bool is_agg,
                         const std::map<int, const TableFragments*>& selected_tables_fragments,
                         const size_t context_count,
                         std::condition_variable& scheduler_cv,
                         std::mutex& scheduler_mutex,
                         std::unordered_set<int>& available_gpus,
                         int& available_cpus);
  std::vector<const int8_t*> fetchIterTabFrags(const size_t frag_id,
                                               const ExecutionDispatch& execution_dispatch,
                                               const InputDescriptor& table_desc,
                                               const int device_id);

  std::pair<std::vector<std::vector<const int8_t*>>, std::vector<std::vector<const int8_t*>>> fetchChunks(
      const ExecutionDispatch&,
      const RelAlgExecutionUnit& ra_exe_unit,
      const int device_id,
      const Data_Namespace::MemoryLevel,
      const std::map<int, const TableFragments*>&,
      const std::map<int, std::vector<size_t>>& selected_fragments,
      const Catalog_Namespace::Catalog&,
      std::list<ChunkIter>&,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>&);

  void buildSelectedFragsMapping(std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
                                 std::vector<size_t>& local_col_to_frag_pos,
                                 const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
                                 const std::map<int, std::vector<size_t>>& selected_fragments,
                                 const std::vector<InputDescriptor>& input_descs);

  RowSetPtr executeResultPlan(const Planner::Result* result_plan,
                              const bool hoist_literals,
                              const ExecutorDeviceType device_type,
                              const ExecutorOptLevel,
                              const Catalog_Namespace::Catalog&,
                              size_t& max_groups_buffer_entry_guess,
                              int32_t* error_code,
                              const Planner::Sort* sort_plan,
                              const bool allow_multifrag,
                              const bool just_explain,
                              const bool allow_loop_joins);
  RowSetPtr executeSortPlan(const Planner::Sort* sort_plan,
                            const int64_t limit,
                            const int64_t offset,
                            const bool hoist_literals,
                            const ExecutorDeviceType device_type,
                            const ExecutorOptLevel,
                            const Catalog_Namespace::Catalog&,
                            size_t& max_groups_buffer_entry_guess,
                            int32_t* error_code,
                            const bool allow_multifrag,
                            const bool just_explain,
                            const bool allow_loop_joins);

  int32_t executePlanWithGroupBy(const RelAlgExecutionUnit& ra_exe_unit,
                                 const CompilationResult&,
                                 const bool hoist_literals,
                                 ResultPtr& results,
                                 const ExecutorDeviceType device_type,
                                 std::vector<std::vector<const int8_t*>>& col_buffers,
                                 const std::vector<size_t> outer_tab_frag_ids,
                                 QueryExecutionContext*,
                                 const std::vector<int64_t>& num_rows,
                                 const std::vector<uint64_t>& dev_frag_row_offsets,
                                 Data_Namespace::DataMgr*,
                                 const int device_id,
                                 const int64_t limit,
                                 const bool was_auto_device,
                                 const uint32_t start_rowid,
                                 const uint32_t num_tables,
                                 RenderAllocatorMap* render_allocator_map);
  int32_t executePlanWithoutGroupBy(const RelAlgExecutionUnit& ra_exe_unit,
                                    const CompilationResult&,
                                    const bool hoist_literals,
                                    ResultPtr& results,
                                    const std::vector<Analyzer::Expr*>& target_exprs,
                                    const ExecutorDeviceType device_type,
                                    std::vector<std::vector<const int8_t*>>& col_buffers,
                                    QueryExecutionContext* query_exe_context,
                                    const std::vector<int64_t>& num_rows,
                                    const std::vector<uint64_t>& dev_frag_row_offsets,
                                    Data_Namespace::DataMgr* data_mgr,
                                    const int device_id,
                                    const uint32_t start_rowid,
                                    const uint32_t num_tables,
                                    RenderAllocatorMap* render_allocator_map);
  std::pair<int64_t, int32_t> reduceResults(const SQLAgg agg,
                                            const SQLTypeInfo& ti,
                                            const int64_t agg_init_val,
                                            const int8_t out_byte_width,
                                            const int64_t* out_vec,
                                            const size_t out_vec_sz,
                                            const bool is_group_by);
  static ResultPtr resultsUnion(ExecutionDispatch& execution_dispatch);
  int64_t getJoinHashTablePtr(const ExecutorDeviceType device_type, const int device_id);
  RowSetPtr reduceMultiDeviceResults(const RelAlgExecutionUnit&,
                                     std::vector<std::pair<ResultPtr, std::vector<size_t>>>& all_fragment_results,
                                     std::shared_ptr<RowSetMemoryOwner>,
                                     const QueryMemoryDescriptor&,
                                     const bool output_columnar,
                                     const ExecutorDeviceType dt_for_all) const;
  RowSetPtr reduceMultiDeviceResultSets(std::vector<std::pair<ResultPtr, std::vector<size_t>>>& all_fragment_results,
                                        std::shared_ptr<RowSetMemoryOwner>,
                                        const QueryMemoryDescriptor&) const;
  RowSetPtr reduceSpeculativeTopN(const RelAlgExecutionUnit&,
                                  std::vector<std::pair<ResultPtr, std::vector<size_t>>>& all_fragment_results,
                                  std::shared_ptr<RowSetMemoryOwner>,
                                  const QueryMemoryDescriptor&) const;
  void executeSimpleInsert(const Planner::RootPlan* root_plan);

  CompilationResult compileWorkUnit(const bool render_output,
                                    const std::vector<InputTableInfo>& query_infos,
                                    const RelAlgExecutionUnit& ra_exe_unit,
                                    const CompilationOptions& co,
                                    const ExecutionOptions& eo,
                                    const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                    const bool allow_lazy_fetch,
                                    std::shared_ptr<RowSetMemoryOwner>,
                                    const size_t max_groups_buffer_entry_count,
                                    const size_t small_groups_buffer_entry_count,
                                    const int8_t crt_min_byte_width,
                                    const JoinInfo& join_info,
                                    const bool has_cardinality_estimation);

  void createErrorCheckControlFlow(llvm::Function* query_func);

  void codegenInnerScanNextRow();

  void allocateInnerScansIterators(const std::vector<InputDescriptor>& input_descs);

  JoinInfo chooseJoinType(const std::list<std::shared_ptr<Analyzer::Expr>>&,
                          const std::vector<InputTableInfo>&,
                          const std::list<std::shared_ptr<const InputColDescriptor>>&,
                          const ExecutorDeviceType device_type);

  void nukeOldState(const bool allow_lazy_fetch,
                    const JoinInfo& join_info,
                    const std::vector<InputTableInfo>& query_infos,
                    const std::list<std::shared_ptr<Analyzer::Expr>>& outer_join_quals);
  std::vector<void*> optimizeAndCodegenCPU(llvm::Function*,
                                           llvm::Function*,
                                           std::unordered_set<llvm::Function*>&,
                                           llvm::Module*,
                                           const CompilationOptions&);
  std::vector<void*> optimizeAndCodegenGPU(llvm::Function*,
                                           llvm::Function*,
                                           std::unordered_set<llvm::Function*>&,
                                           llvm::Module*,
                                           const bool no_inline,
                                           const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                           const CompilationOptions&);
  std::string generatePTX(const std::string&) const;
  void initializeNVPTXBackend() const;

  int8_t warpSize() const;
  unsigned gridSize() const;
  unsigned blockSize() const;

  llvm::Value* groupByColumnCodegen(Analyzer::Expr* group_by_col,
                                    const CompilationOptions&,
                                    const bool translate_null_val,
                                    const int64_t translated_null_val,
                                    GroupByAndAggregate::DiamondCodegen&,
                                    std::stack<llvm::BasicBlock*>&,
                                    const bool thread_mem_shared);

  llvm::Value* castToFP(llvm::Value* val);
  llvm::Value* castToTypeIn(llvm::Value* val, const size_t bit_width);
  llvm::Value* castToIntPtrTyIn(llvm::Value* val, const size_t bit_width);

  void allocateLocalColumnIds(const std::list<std::shared_ptr<const InputColDescriptor>>& global_col_ids);
  int getLocalColumnId(const Analyzer::ColumnVar* col_var, const bool fetch_column) const;

  std::pair<bool, int64_t> skipFragment(const InputDescriptor& table_desc,
                                        const Fragmenter_Namespace::FragmentInfo& frag_info,
                                        const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                                        const std::vector<uint64_t>& all_frag_row_offsets,
                                        const size_t frag_idx);

  typedef std::vector<std::string> CodeCacheKey;
  typedef std::vector<std::tuple<void*, std::unique_ptr<llvm::ExecutionEngine>, std::unique_ptr<GpuCompilationContext>>>
      CodeCacheVal;
  std::vector<void*> getCodeFromCache(const CodeCacheKey&,
                                      const std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>&);
  void addCodeToCache(const CodeCacheKey&,
                      const std::vector<std::tuple<void*, llvm::ExecutionEngine*, GpuCompilationContext*>>&,
                      llvm::Module*,
                      std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>&);

  std::vector<int8_t> serializeLiterals(const std::unordered_map<int, Executor::LiteralValues>& literals,
                                        const int device_id);

  static size_t literalBytes(const LiteralValue& lit) {
    switch (lit.which()) {
      case 0:
        return 1;
      case 1:
        return 2;
      case 2:
        return 4;
      case 3:
        return 8;
      case 4:
        return 4;
      case 5:
        return 8;
      case 6:
        return 4;
      case 7:
        return 4;
      default:
        abort();
    }
  }

  static size_t addAligned(const size_t off_in, const size_t alignment) {
    size_t off = off_in;
    if (off % alignment != 0) {
      off += (alignment - off % alignment);
    }
    return off + alignment;
  }

  struct CgenState {
   public:
    CgenState(const std::vector<InputTableInfo>& query_infos, const bool is_outer_join)
        : module_(nullptr),
          row_func_(nullptr),
          context_(getGlobalLLVMContext()),
          ir_builder_(context_),
          is_outer_join_(is_outer_join),
          outer_join_cond_lv_(nullptr),
          query_infos_(query_infos),
          must_run_on_cpu_(false),
          needs_error_check_(false) {}

    size_t getOrAddLiteral(const Analyzer::Constant* constant,
                           const EncodingType enc_type,
                           const int dict_id,
                           const int device_id) {
      const auto& ti = constant->get_type_info();
      const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
      switch (type) {
        case kBOOLEAN:
          return getOrAddLiteral(constant->get_is_null() ? int8_t(inline_int_null_val(ti))
                                                         : int8_t(constant->get_constval().boolval ? 1 : 0),
                                 device_id);
        case kSMALLINT:
          return getOrAddLiteral(
              constant->get_is_null() ? int16_t(inline_int_null_val(ti)) : constant->get_constval().smallintval,
              device_id);
        case kINT:
          return getOrAddLiteral(
              constant->get_is_null() ? int32_t(inline_int_null_val(ti)) : constant->get_constval().intval, device_id);
        case kBIGINT:
          return getOrAddLiteral(
              constant->get_is_null() ? int64_t(inline_int_null_val(ti)) : constant->get_constval().bigintval,
              device_id);
        case kFLOAT:
          return getOrAddLiteral(
              constant->get_is_null() ? float(inline_fp_null_val(ti)) : constant->get_constval().floatval, device_id);
        case kDOUBLE:
          return getOrAddLiteral(constant->get_is_null() ? inline_fp_null_val(ti) : constant->get_constval().doubleval,
                                 device_id);
        case kCHAR:
        case kTEXT:
        case kVARCHAR:
          if (enc_type == kENCODING_DICT) {
            if (constant->get_is_null()) {
              return getOrAddLiteral(int32_t(inline_int_null_val(ti)), device_id);
            }
            return getOrAddLiteral(std::make_pair(*constant->get_constval().stringval, dict_id), device_id);
          }
          CHECK_EQ(kENCODING_NONE, enc_type);
          if (constant->get_is_null()) {
            throw std::runtime_error(
                "CHAR / VARCHAR NULL literal not supported in this context");  // TODO(alex): support null
          }
          return getOrAddLiteral(*constant->get_constval().stringval, device_id);
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
        case kINTERVAL_DAY_TIME:
        case kINTERVAL_YEAR_MONTH:
          // TODO(alex): support null
          return getOrAddLiteral(static_cast<int64_t>(constant->get_constval().timeval), device_id);
        default:
          abort();
      }
    }

    const std::unordered_map<int, LiteralValues>& getLiterals() const { return literals_; }

    llvm::Value* addStringConstant(const std::string& str) {
      llvm::Value* str_lv =
          ir_builder_.CreateGlobalString(str, "str_const_" + std::to_string(std::hash<std::string>()(str)));
      auto i8_ptr = llvm::PointerType::get(get_int_type(8, context_), 0);
      str_constants_.push_back(str_lv);
      str_lv = ir_builder_.CreateBitCast(str_lv, i8_ptr);
      return str_lv;
    }

    const InValuesBitmap* addInValuesBitmap(std::unique_ptr<InValuesBitmap>& in_values_bitmap) {
      in_values_bitmaps_.emplace_back(std::move(in_values_bitmap));
      return in_values_bitmaps_.back().get();
    }
    // look up a runtime function based on the name, return type and type of
    // the arguments and call it; x64 only, don't call from GPU codegen
    llvm::Value* emitExternalCall(const std::string& fname,
                                  llvm::Type* ret_type,
                                  const std::vector<llvm::Value*> args) {
      std::vector<llvm::Type*> arg_types;
      for (const auto arg : args) {
        arg_types.push_back(arg->getType());
      }
      auto func_ty = llvm::FunctionType::get(ret_type, arg_types, false);
      auto func_p = module_->getOrInsertFunction(fname, func_ty);
      CHECK(func_p);
      llvm::Value* result = ir_builder_.CreateCall(func_p, args);
      // check the assumed type
      CHECK_EQ(result->getType(), ret_type);
      return result;
    }

    llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args) {
      auto f = module_->getFunction(fname);
      CHECK(f);
      return ir_builder_.CreateCall(f, args);
    }

    llvm::Module* module_;
    llvm::Function* row_func_;
    std::vector<llvm::Function*> helper_functions_;
    llvm::LLVMContext& context_;
    llvm::IRBuilder<> ir_builder_;
    std::unordered_map<int, std::vector<llvm::Value*>> fetch_cache_;
    std::vector<llvm::Value*> group_by_expr_cache_;
    std::vector<llvm::Value*> str_constants_;
    std::unordered_map<InputDescriptor, std::pair<llvm::Value*, llvm::Value*>> scan_to_iterator_;
    const bool is_outer_join_;
    llvm::Value* outer_join_cond_lv_;
    std::vector<llvm::BasicBlock*> inner_scan_labels_;
    std::unordered_map<int, llvm::Value*> scan_idx_to_hash_pos_;
    std::vector<std::unique_ptr<const InValuesBitmap>> in_values_bitmaps_;
    const std::vector<InputTableInfo>& query_infos_;
    bool must_run_on_cpu_;
    bool needs_error_check_;

   private:
    template <class T>
    size_t getOrAddLiteral(const T& val, const int device_id) {
      const Executor::LiteralValue var_val(val);
      size_t literal_found_off{0};
      auto& literals = literals_[device_id];
      for (const auto& literal : literals) {
        const auto lit_bytes = literalBytes(literal);
        literal_found_off = addAligned(literal_found_off, lit_bytes);
        if (literal == var_val) {
          return literal_found_off - lit_bytes;
        }
      }
      literals.emplace_back(val);
      const auto lit_bytes = literalBytes(var_val);
      literal_bytes_[device_id] = addAligned(literal_bytes_[device_id], lit_bytes);
      return literal_bytes_[device_id] - lit_bytes;
    }

    std::unordered_map<int, LiteralValues> literals_;
    std::unordered_map<int, size_t> literal_bytes_;
  };
  std::unique_ptr<CgenState> cgen_state_;

  class FetchCacheAnchor {
   public:
    FetchCacheAnchor(CgenState* cgen_state) : cgen_state_(cgen_state), saved_fetch_cache(cgen_state_->fetch_cache_) {}
    ~FetchCacheAnchor() { cgen_state_->fetch_cache_.swap(saved_fetch_cache); }

   private:
    CgenState* cgen_state_;
    std::unordered_map<int, std::vector<llvm::Value*>> saved_fetch_cache;
  };

  // TODO(alex): remove, only useful for the legacy path
  class ResetIsNested {
   public:
    ResetIsNested(Executor* executor) : executor_(executor) {}
    ~ResetIsNested() { executor_->is_nested_ = false; }

   private:
    Executor* executor_;
  };

  struct PlanState {
    PlanState(const bool allow_lazy_fetch, const JoinInfo& join_info, const Executor* executor)
        : allow_lazy_fetch_(allow_lazy_fetch), join_info_(join_info), executor_(executor) {}

    std::vector<int64_t> init_agg_vals_;
    std::vector<Analyzer::Expr*> target_exprs_;
    std::unordered_map<InputColDescriptor, int> global_to_local_col_ids_;
    std::vector<int> local_to_global_col_ids_;
    std::set<std::pair<int, int>> columns_to_fetch_;
    std::set<std::pair<int, int>> columns_to_not_fetch_;
    bool allow_lazy_fetch_;
    JoinInfo join_info_;
    const Executor* executor_;

    bool isLazyFetchColumn(const Analyzer::Expr* target_expr) {
      if (!allow_lazy_fetch_) {
        return false;
      }
      const auto do_not_fetch_column = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      if (!do_not_fetch_column || dynamic_cast<const Analyzer::Var*>(do_not_fetch_column)) {
        return false;
      }
      if (do_not_fetch_column->get_table_id() > 0) {
        auto cd = get_column_descriptor(
            do_not_fetch_column->get_column_id(), do_not_fetch_column->get_table_id(), *executor_->catalog_);
        if (cd->isVirtualCol) {
          return false;
        }
      }
      std::set<std::pair<int, int>> intersect;
      std::set_intersection(columns_to_fetch_.begin(),
                            columns_to_fetch_.end(),
                            columns_to_not_fetch_.begin(),
                            columns_to_not_fetch_.end(),
                            std::inserter(intersect, intersect.begin()));
      if (!intersect.empty()) {
        throw CompilationRetryNoLazyFetch();
      }
      return columns_to_fetch_.find(std::make_pair(do_not_fetch_column->get_table_id(),
                                                   do_not_fetch_column->get_column_id())) == columns_to_fetch_.end();
    }
  };

  std::unordered_set<llvm::Function*> markDeadRuntimeFuncs(llvm::Module& module,
                                                           const std::vector<llvm::Function*>& roots,
                                                           const std::vector<llvm::Function*>& leaves);

  llvm::Value* spillDoubleElement(llvm::Value* elem_val, llvm::Type* elem_ty);

  std::unique_ptr<PlanState> plan_state_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  bool is_nested_;

  static const int max_gpu_count{16};
  std::mutex gpu_exec_mutex_[max_gpu_count];

  mutable std::shared_ptr<StringDictionary> lit_str_dict_;
  mutable std::mutex str_dict_mutex_;

  mutable std::unique_ptr<llvm::TargetMachine> nvptx_target_machine_;

  std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>> cpu_code_cache_;
  std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>> gpu_code_cache_;

  ::QueryRenderer::QueryRenderManager* render_manager_;

  const size_t small_groups_buffer_entry_count_{512};
  const size_t render_small_groups_buffer_entry_count_{2 * 1024 * 1024};

  const unsigned block_size_x_;
  const unsigned grid_size_x_;
  const std::string debug_dir_;
  const std::string debug_file_;

  const int db_id_;
  const Catalog_Namespace::Catalog* catalog_;
  const TemporaryTables* temporary_tables_;

  InputTableInfoCache input_table_info_cache_;

  static std::map<std::pair<int, ::QueryRenderer::QueryRenderManager*>, std::shared_ptr<Executor>> executors_;
  static std::mutex execute_mutex_;
  static mapd_shared_mutex executors_cache_mutex_;

  static const int32_t ERR_DIV_BY_ZERO{1};
  static const int32_t ERR_OUT_OF_GPU_MEM{2};
  static const int32_t ERR_OUT_OF_SLOTS{3};
  static const int32_t ERR_UNSUPPORTED_SELF_JOIN{4};
  static const int32_t ERR_OUT_OF_RENDER_MEM{5};
  static const int32_t ERR_OUT_OF_CPU_MEM{6};
  static const int32_t ERR_OVERFLOW_OR_UNDERFLOW{7};
  static const int32_t ERR_SPECULATIVE_TOP_OOM{8};
  friend class GroupByAndAggregate;
  friend struct QueryMemoryDescriptor;
  friend class QueryExecutionContext;
  friend class ResultRows;
  friend class ResultSet;
  friend class IteratorTable;
  friend class InValuesBitmap;
  friend class JoinHashTable;
  friend class LeafAggregator;
  friend class QueryRewriter;
  friend class RelAlgExecutor;
  friend class ExecutionDispatch;
};

#endif  // QUERYENGINE_EXECUTE_H
