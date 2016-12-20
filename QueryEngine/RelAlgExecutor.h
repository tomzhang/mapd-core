#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "InputMetadata.h"
#include "Execute.h"
#include "QueryRewrite.h"
#include "RelAlgExecutionDescriptor.h"
#include "SpeculativeTopN.h"
#include "../LeafAggregator.h"

#include <ctime>

struct FirstStepExecutionResult {
  ExecutionResult result;
  const unsigned node_id;
  bool is_outermost_query;
};

class RelAlgExecutor {
 public:
  RelAlgExecutor(Executor* executor, const Catalog_Namespace::Catalog& cat) : executor_(executor), cat_(cat), now_(0) {}

  ExecutionResult executeRelAlgQuery(const std::string& query_ra,
                                     const CompilationOptions& co,
                                     const ExecutionOptions& eo,
                                     RenderInfo* render_info);

  FirstStepExecutionResult executeRelAlgQueryFirstStep(const RelAlgNode* ra,
                                                       const CompilationOptions& co,
                                                       const ExecutionOptions& eo,
                                                       RenderInfo* render_info);

  const std::vector<std::string>& getScanTableNamesInRelAlgSeq() const;

  void addLeafResult(const unsigned id, const AggregatedResult& result) {
    const auto it_ok = leaf_results_.emplace(id, result);
    CHECK(it_ok.second);
  }

  void registerSubquery(RexSubQuery* subquery) noexcept { subqueries_.push_back(subquery); }

  const std::vector<RexSubQuery*>& getSubqueries() const noexcept { return subqueries_; };

 private:
  ExecutionResult executeRelAlgSubQuery(const RelAlgNode* subquery_ra,
                                        const CompilationOptions& co,
                                        const ExecutionOptions& eo,
                                        RenderInfo* render_info,
                                        const int64_t queue_time_ms);

  ExecutionResult executeRelAlgSeq(std::vector<RaExecutionDesc>& ed_list,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   RenderInfo* render_info,
                                   const int64_t queue_time_ms);

  void executeRelAlgStep(const size_t step_idx,
                         std::vector<RaExecutionDesc>&,
                         const CompilationOptions&,
                         const ExecutionOptions&,
                         RenderInfo*,
                         const int64_t queue_time_ms);

  ExecutionResult executeCompound(const RelCompound*,
                                  const CompilationOptions&,
                                  const ExecutionOptions&,
                                  RenderInfo*,
                                  const int64_t queue_time_ms);

  ExecutionResult executeAggregate(const RelAggregate* aggregate,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   RenderInfo* render_info,
                                   const int64_t queue_time_ms);

  ExecutionResult executeProject(const RelProject*,
                                 const CompilationOptions&,
                                 const ExecutionOptions&,
                                 RenderInfo*,
                                 const int64_t queue_time_ms);

  ExecutionResult executeFilter(const RelFilter*,
                                const CompilationOptions&,
                                const ExecutionOptions&,
                                RenderInfo*,
                                const int64_t queue_time_ms);

  ExecutionResult executeSort(const RelSort*,
                              const CompilationOptions&,
                              const ExecutionOptions&,
                              RenderInfo*,
                              const int64_t queue_time_ms);

  ExecutionResult executeJoin(const RelJoin*,
                              const CompilationOptions&,
                              const ExecutionOptions&,
                              RenderInfo*,
                              const int64_t queue_time_ms);

  // TODO(alex): just move max_groups_buffer_entry_guess to RelAlgExecutionUnit once
  //             we deprecate the plan-based executor paths and remove WorkUnit
  struct WorkUnit {
    RelAlgExecutionUnit exe_unit;
    const RelAlgNode* body;
    const size_t max_groups_buffer_entry_guess;
    std::unique_ptr<QueryRewriter> query_rewriter;
  };

  WorkUnit createSortInputWorkUnit(const RelSort*, const bool just_explain);

  ExecutionResult executeWorkUnit(const WorkUnit& work_unit,
                                  const std::vector<TargetMetaInfo>& targets_meta,
                                  const bool is_agg,
                                  const CompilationOptions& co,
                                  const ExecutionOptions& eo,
                                  RenderInfo*,
                                  const int64_t queue_time_ms);

  size_t getNDVEstimation(const WorkUnit& work_unit,
                          const bool is_agg,
                          const CompilationOptions& co,
                          const ExecutionOptions& eo);

  ssize_t getFilteredCountAll(const WorkUnit& work_unit,
                              const bool is_agg,
                              const CompilationOptions& co,
                              const ExecutionOptions& eo);

  bool isRowidLookup(const WorkUnit& work_unit);

  ExecutionResult renderWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                 const std::vector<TargetMetaInfo>& targets_meta,
                                 RenderInfo* render_info,
                                 const int32_t error_code,
                                 const int64_t queue_time_ms);

  ExecutionResult handleRetry(const int32_t error_code_in,
                              const RelAlgExecutor::WorkUnit& work_unit,
                              const std::vector<TargetMetaInfo>& targets_meta,
                              const bool is_agg,
                              const CompilationOptions& co,
                              const ExecutionOptions& eo,
                              const int64_t queue_time_ms);

  WorkUnit createWorkUnit(const RelAlgNode*, const SortInfo&, const bool just_explain);

  WorkUnit createCompoundWorkUnit(const RelCompound*, const SortInfo&, const bool just_explain);

  WorkUnit createAggregateWorkUnit(const RelAggregate*, const SortInfo&, const bool just_explain);

  WorkUnit createProjectWorkUnit(const RelProject*, const SortInfo&, const bool just_explain);

  WorkUnit createFilterWorkUnit(const RelFilter*, const SortInfo&, const bool just_explain);

  WorkUnit createJoinWorkUnit(const RelJoin*, const SortInfo&, const bool just_explain);

  void addTemporaryTable(const int table_id, const ResultPtr& result) {
    auto row_set = boost::get<RowSetPtr>(&result);
    if (row_set) {
      CHECK_LT(size_t(0), (*row_set)->colCount());
    }
    CHECK_LT(table_id, 0);
    const auto it_ok = temporary_tables_.emplace(table_id, result);
    CHECK(it_ok.second);
  }

  void handleNop(const RelAlgNode*);

  static std::vector<std::string> getScanTableNamesInRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs);

  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
  TemporaryTables temporary_tables_;
  time_t now_;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_;  // TODO(alex): remove
  std::vector<std::string> table_names_;  // used by poly rendering only, lazily initialized by executeRelAlgQuery()
  std::vector<RexSubQuery*> subqueries_;
  std::unordered_map<unsigned, AggregatedResult> leaf_results_;
  static SpeculativeTopNBlacklist speculative_topn_blacklist_;
  static const size_t max_groups_buffer_entry_default_guess{16384};
};

#endif  // QUERYENGINE_RELALGEXECUTOR_H
