#ifdef HAVE_CALCITE
#include "RelAlgExecutor.h"
#include "RelAlgTranslator.h"

#include "CardinalityEstimator.h"
#include "InputMetadata.h"
#include "RexVisitor.h"
#include "ScalarExprVisitor.h"

#include "../Shared/measure.h"

ExecutionResult RelAlgExecutor::executeRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo,
                                                 RenderInfo* render_info) {
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);
  int64_t queue_time_ms = timer_stop(clock_begin);
  Executor::RowSetHolder row_set_holder(executor_);
  decltype(temporary_tables_)().swap(temporary_tables_);
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;
  time(&now_);
  CHECK(!exec_descs.empty());
  const auto exec_desc_count = eo.just_explain ? size_t(1) : exec_descs.size();
  for (size_t i = 0; i < exec_desc_count; ++i) {
    auto& exec_desc = exec_descs[i];
    const auto body = exec_desc.getBody();
    if (body->isNop()) {
      handleNop(body);
      continue;
    }
    const ExecutionOptions eo_work_unit{eo.output_columnar_hint,
                                        eo.allow_multifrag,
                                        eo.just_explain,
                                        eo.allow_loop_joins,
                                        eo.with_watchdog && (i == 0 || dynamic_cast<const RelProject*>(body))};
    const auto compound = dynamic_cast<const RelCompound*>(body);
    if (compound) {
      exec_desc.setResult(executeCompound(compound, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-compound->getId(), exec_desc.getResult().getDataPtr());
      continue;
    }
    const auto project = dynamic_cast<const RelProject*>(body);
    if (project) {
      exec_desc.setResult(executeProject(project, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-project->getId(), exec_desc.getResult().getDataPtr());
      continue;
    }
    const auto aggregate = dynamic_cast<const RelAggregate*>(body);
    if (aggregate) {
      exec_desc.setResult(executeAggregate(aggregate, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-aggregate->getId(), exec_desc.getResult().getDataPtr());
      continue;
    }
    const auto filter = dynamic_cast<const RelFilter*>(body);
    if (filter) {
      exec_desc.setResult(executeFilter(filter, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-filter->getId(), exec_desc.getResult().getDataPtr());
      continue;
    }
    const auto sort = dynamic_cast<const RelSort*>(body);
    if (sort) {
      exec_desc.setResult(executeSort(sort, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-sort->getId(), exec_desc.getResult().getDataPtr());
      continue;
    }
#ifdef ENABLE_JOIN_EXEC
    const auto join = dynamic_cast<const RelJoin*>(body);
    if (join) {
      exec_desc.setResult(executeJoin(join, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-join->getId(), exec_desc.getResult().getDataPtr());
      continue;
    }
#endif
    CHECK(false);
  }
  return exec_descs[exec_desc_count - 1].getResult();
}

std::vector<TargetMetaInfo> RelAlgExecutor::validateRelAlgSeq(const std::vector<RaExecutionDesc>& exec_descs) {
  CHECK(!exec_descs.empty());
  for (const auto& exec_desc : exec_descs) {
    const auto body = exec_desc.getBody();
    const auto result = exec_desc.getResult();
    CHECK(result.empty());
    if (body->isNop()) {
      CHECK(dynamic_cast<const RelAggregate*>(body));
      CHECK_EQ(size_t(1), body->inputCount());
      const auto input = body->getInput(0);
      body->setOutputMetainfo(input->getOutputMetainfo());
      addTemporaryTable(-body->getId(), result.getDataPtr());
      continue;
    }
    const auto compound = dynamic_cast<const RelCompound*>(body);
    SortInfo dummy{{}, SortAlgorithm::Default, 0, 0};
    if (compound) {
      createCompoundWorkUnit(compound, dummy);
      addTemporaryTable(-compound->getId(), result.getDataPtr());
      continue;
    }
    const auto project = dynamic_cast<const RelProject*>(body);
    if (project) {
      createProjectWorkUnit(project, dummy);
      addTemporaryTable(-project->getId(), result.getDataPtr());
      continue;
    }
    const auto filter = dynamic_cast<const RelFilter*>(body);
    if (filter) {
      createFilterWorkUnit(filter, dummy);
      addTemporaryTable(-filter->getId(), result.getDataPtr());
      continue;
    }
    const auto sort = dynamic_cast<const RelSort*>(body);
    if (sort) {
      CHECK_EQ(size_t(1), sort->inputCount());
      const auto source = sort->getInput(0);
      CHECK(!dynamic_cast<const RelSort*>(source));
      createSortInputWorkUnit(sort);
      addTemporaryTable(-sort->getId(), result.getDataPtr());
      continue;
    }
  }
  return exec_descs.back().getBody()->getOutputMetainfo();
}

std::vector<std::string> RelAlgExecutor::getScanTableNamesInRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs) {
  if (exec_descs.empty()) {
    return {};
  }
  std::unordered_set<std::string> table_names;
  std::unordered_set<const RelAlgNode*> visited;
  std::vector<const RelAlgNode*> work_set;
  for (const auto& exec_desc : exec_descs) {
    const auto body = exec_desc.getBody();
    if (visited.count(body)) {
      continue;
    }
    work_set.push_back(body);
    while (!work_set.empty()) {
      auto walker = work_set.back();
      work_set.pop_back();
      if (visited.count(walker)) {
        continue;
      }
      visited.insert(walker);
      if (walker->isNop()) {
        CHECK_EQ(size_t(1), walker->inputCount());
        work_set.push_back(walker->getInput(0));
        continue;
      }
      if (const auto scan = dynamic_cast<const RelScan*>(walker)) {
        auto td = scan->getTableDescriptor();
        CHECK(td);
        table_names.insert(td->tableName);
        continue;
      }
      const auto compound = dynamic_cast<const RelCompound*>(walker);
      const auto join = dynamic_cast<const RelJoin*>(walker);
      const auto project = dynamic_cast<const RelProject*>(walker);
      const auto aggregate = dynamic_cast<const RelAggregate*>(walker);
      const auto filter = dynamic_cast<const RelFilter*>(walker);
      const auto sort = dynamic_cast<const RelSort*>(walker);
      if (compound || join || project || aggregate || filter || sort) {
        for (size_t i = 0; i < walker->inputCount(); ++i) {
          work_set.push_back(walker->getInput(i));
        }
        continue;
      }
      CHECK(false);
    }
  }
  return std::vector<std::string>(table_names.begin(), table_names.end());
}

void RelAlgExecutor::handleNop(const RelAlgNode* body) {
  CHECK(dynamic_cast<const RelAggregate*>(body));
  CHECK_EQ(size_t(1), body->inputCount());
  const auto input = body->getInput(0);
  body->setOutputMetainfo(input->getOutputMetainfo());
  const auto it = temporary_tables_.find(-input->getId());
  CHECK(it != temporary_tables_.end());
  addTemporaryTable(-body->getId(), it->second);
}

namespace {

class RexUsedInputsVisitor : public RexVisitor<std::unordered_set<const RexInput*>> {
 public:
  std::unordered_set<const RexInput*> visitInput(const RexInput* rex_input) const override { return {rex_input}; }

 protected:
  std::unordered_set<const RexInput*> aggregateResult(
      const std::unordered_set<const RexInput*>& aggregate,
      const std::unordered_set<const RexInput*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

const RelAlgNode* get_data_sink(const RelAlgNode* ra_node) {
  const bool is_join = dynamic_cast<const RelJoin*>(ra_node) != nullptr;
  CHECK((is_join && 2 == ra_node->inputCount()) || (!is_join && 1 == ra_node->inputCount()));
  const auto join_input =
      is_join ? static_cast<const RelJoin*>(ra_node) : dynamic_cast<const RelJoin*>(ra_node->getInput(0));
  // If the input node is a join, the data is sourced from it instead of the initial node.
  const auto data_sink_node =
      join_input ? static_cast<const RelAlgNode*>(join_input) : static_cast<const RelAlgNode*>(ra_node);
  CHECK(1 <= data_sink_node->inputCount() && data_sink_node->inputCount() <= 2);
  return data_sink_node;
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelCompound* compound) {
  RexUsedInputsVisitor visitor;
  const auto filter_expr = compound->getFilterExpr();
  std::unordered_set<const RexInput*> used_inputs =
      filter_expr ? visitor.visit(filter_expr) : std::unordered_set<const RexInput*>{};
  const auto sources_size = compound->getScalarSourcesSize();
  for (size_t i = 0; i < sources_size; ++i) {
    const auto source_inputs = visitor.visit(compound->getScalarSource(i));
    used_inputs.insert(source_inputs.begin(), source_inputs.end());
  }
  return std::make_pair(used_inputs, std::vector<std::shared_ptr<RexInput>>{});
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelAggregate* aggregate) {
  CHECK_EQ(size_t(1), aggregate->inputCount());
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto source = aggregate->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  const auto group_count = aggregate->getGroupByCount();
  CHECK_GE(in_metainfo.size(), group_count);
  for (size_t i = 0; i < group_count; ++i) {
    auto synthesized_used_input = new RexInput(source, i);
    used_inputs_owned.emplace_back(synthesized_used_input);
    used_inputs.insert(synthesized_used_input);
  }
  for (const auto& agg_expr : aggregate->getAggExprs()) {
    const auto operand_idx = agg_expr->getOperand();
    const bool takes_arg{operand_idx >= 0};
    if (takes_arg) {
      CHECK_GE(in_metainfo.size(), static_cast<size_t>(operand_idx));
      auto synthesized_used_input = new RexInput(source, operand_idx);
      used_inputs_owned.emplace_back(synthesized_used_input);
      used_inputs.insert(synthesized_used_input);
    }
  }
  return std::make_pair(used_inputs, used_inputs_owned);
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelProject* project) {
  RexUsedInputsVisitor visitor;
  std::unordered_set<const RexInput*> used_inputs;
  for (size_t i = 0; i < project->size(); ++i) {
    const auto proj_inputs = visitor.visit(project->getProjectAt(i));
    used_inputs.insert(proj_inputs.begin(), proj_inputs.end());
  }
  return std::make_pair(used_inputs, std::vector<std::shared_ptr<RexInput>>{});
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelFilter* filter) {
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto data_sink_node = get_data_sink(filter);
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto source = data_sink_node->getInput(nest_level);
    const auto scan_source = dynamic_cast<const RelScan*>(source);
    if (scan_source) {
      CHECK(source->getOutputMetainfo().empty());
      for (size_t i = 0; i < scan_source->size(); ++i) {
        auto synthesized_used_input = new RexInput(scan_source, i);
        used_inputs_owned.emplace_back(synthesized_used_input);
        used_inputs.insert(synthesized_used_input);
      }
    } else {
      const auto& partial_in_metadata = source->getOutputMetainfo();
      for (size_t i = 0; i < partial_in_metadata.size(); ++i) {
        auto synthesized_used_input = new RexInput(source, i);
        used_inputs_owned.emplace_back(synthesized_used_input);
        used_inputs.insert(synthesized_used_input);
      }
    }
  }
  return std::make_pair(used_inputs, used_inputs_owned);
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelJoin* join) {
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto lhs = join->getInput(0);
  if (dynamic_cast<const RelJoin*>(lhs)) {
    auto synthesized_used_input = new RexInput(lhs, 0);
    used_inputs_owned.emplace_back(synthesized_used_input);
    used_inputs.insert(synthesized_used_input);
    for (auto previous_join = static_cast<const RelJoin*>(lhs); previous_join;
         previous_join = dynamic_cast<const RelJoin*>(previous_join->getInput(0))) {
      synthesized_used_input = new RexInput(lhs, 0);
      used_inputs_owned.emplace_back(synthesized_used_input);
      used_inputs.insert(synthesized_used_input);
    }
  }
  return std::make_pair(used_inputs, used_inputs_owned);
}

int table_id_from_ra(const RelAlgNode* ra_node) {
  const auto scan_ra = dynamic_cast<const RelScan*>(ra_node);
  if (scan_ra) {
    const auto td = scan_ra->getTableDescriptor();
    CHECK(td);
    return td->tableId;
  }
  return -ra_node->getId();
}

std::unordered_map<const RelAlgNode*, int> get_input_nest_levels(const RelAlgNode* ra_node) {
  const auto data_sink_node = get_data_sink(ra_node);
  std::unordered_map<const RelAlgNode*, int> input_to_nest_level;
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto input_ra = data_sink_node->getInput(nest_level);
    const auto it_ok = input_to_nest_level.emplace(input_ra, nest_level);
    CHECK(it_ok.second);
  }
  return input_to_nest_level;
}

std::unordered_set<const RexInput*> get_join_source_used_inputs(const RelAlgNode* ra_node) {
  const bool is_join = dynamic_cast<const RelJoin*>(ra_node) != nullptr;
  CHECK((is_join && 2 == ra_node->inputCount()) || (!is_join && 1 == ra_node->inputCount()));
  const auto join_input =
      is_join ? static_cast<const RelJoin*>(ra_node) : dynamic_cast<const RelJoin*>(ra_node->getInput(0));
  if (join_input) {
    const auto join_cond = join_input->getCondition();
    RexUsedInputsVisitor visitor;
    return visitor.visit(join_cond);
  }
  return std::unordered_set<const RexInput*>{};
}

size_t get_target_list_size(const RelAlgNode* ra_node) {
  const auto scan = dynamic_cast<const RelScan*>(ra_node);
  if (scan) {
    return scan->getFieldNames().size();
  }
  const auto join = dynamic_cast<const RelJoin*>(ra_node);
  if (join) {
    return get_target_list_size(join->getInput(0)) + get_target_list_size(join->getInput(1));
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(ra_node);
  if (aggregate) {
    return aggregate->getFields().size();
  }
  const auto compound = dynamic_cast<const RelCompound*>(ra_node);
  if (compound) {
    return compound->getFields().size();
  }
  const auto filter = dynamic_cast<const RelFilter*>(ra_node);
  if (filter) {
    return get_target_list_size(filter->getInput(0));
  }
  const auto project = dynamic_cast<const RelProject*>(ra_node);
  if (project) {
    return project->getFields().size();
  }
  const auto sort = dynamic_cast<const RelSort*>(ra_node);
  if (sort) {
    return get_target_list_size(sort->getInput(0));
  }
  CHECK(false);
  return 0;
}

std::vector<const RelAlgNode*> get_join_sequence(const RelAlgNode* ra) {
  std::vector<const RelAlgNode*> seq;
  for (auto join = dynamic_cast<const RelJoin*>(ra); join; join = static_cast<const RelJoin*>(join->getInput(0))) {
    CHECK_EQ(size_t(2), join->inputCount());
    seq.emplace_back(join->getInput(1));
    auto lhs = join->getInput(0);
    if (!dynamic_cast<const RelJoin*>(lhs)) {
      seq.emplace_back(lhs);
      break;
    }
  }
  std::reverse(seq.begin(), seq.end());
  return seq;
}

std::pair<const RelAlgNode*, int> get_non_join_source_node(const RelAlgNode* crt_source, const int col_id) {
  CHECK_LE(0, col_id);
  const auto join = dynamic_cast<const RelJoin*>(crt_source);
  if (!join) {
    return std::make_pair(crt_source, col_id);
  }
  const auto lhs = join->getInput(0);
  const auto rhs = join->getInput(1);
  const size_t left_source_size = get_target_list_size(lhs);
  if (size_t(col_id) >= left_source_size) {
    return std::make_pair(rhs, col_id - int(left_source_size));
  }
  if (dynamic_cast<const RelJoin*>(lhs)) {
    return get_non_join_source_node(static_cast<const RelJoin*>(lhs), col_id);
  }
  return std::make_pair(lhs, col_id);
}

void collect_used_input_desc(std::vector<InputDescriptor>& input_descs,
                             std::unordered_set<std::shared_ptr<const InputColDescriptor>>& input_col_descs_unique,
                             const RelAlgNode* ra_node,
                             const std::unordered_set<const RexInput*>& source_used_inputs,
                             const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::unordered_set<InputDescriptor> input_descs_unique(input_descs.begin(), input_descs.end());
  const auto join_seq = get_join_sequence(get_data_sink(ra_node));
  std::unordered_map<const RelAlgNode*, int> non_join_to_nest_level;
  for (const auto node : join_seq) {
    non_join_to_nest_level.insert(std::make_pair(node, non_join_to_nest_level.size()));
  }
  for (const auto used_input : source_used_inputs) {
    const auto input_ra = used_input->getSourceNode();
    const int table_id = table_id_from_ra(input_ra);
    const auto col_id = used_input->getIndex();
    auto it = input_to_nest_level.find(input_ra);
    if (it == input_to_nest_level.end()) {
      throw std::runtime_error("Multi-way join not supported");
    }
    const int input_desc = it->second;

    const RelAlgNode* indirect_input_ra{nullptr};
    int indirect_col_id{-1};
    std::tie(indirect_input_ra, indirect_col_id) = get_non_join_source_node(input_ra, col_id);
    if (indirect_input_ra == input_ra) {
      CHECK_EQ(indirect_col_id, static_cast<ssize_t>(col_id));
      input_col_descs_unique.insert(std::make_shared<const InputColDescriptor>(
          dynamic_cast<const RelScan*>(input_ra) ? col_id + 1 : col_id, table_id, input_desc));
      continue;
    }

    // A column from indirect source indexed by an iterator
    const int indirect_table_id = table_id_from_ra(indirect_input_ra);
    CHECK(!input_to_nest_level.count(indirect_input_ra));
    it = non_join_to_nest_level.find(indirect_input_ra);
    CHECK(it != non_join_to_nest_level.end());
    const int nest_level = it->second;
    if (!input_descs_unique.count(InputDescriptor(indirect_table_id, -1))) {
      input_descs_unique.emplace(indirect_table_id, -1);
      input_descs.emplace_back(indirect_table_id, -1);
    }
    CHECK(!dynamic_cast<const RelScan*>(input_ra));
    CHECK_EQ(size_t(0), static_cast<size_t>(input_desc));
    // Physical columns from a scan node are numbered from 1 in our system.
    input_col_descs_unique.insert(std::make_shared<const IndirectInputColDescriptor>(
        col_id,
        table_id,
        input_desc,
        nest_level,
        table_id,
        input_desc,
        dynamic_cast<const RelScan*>(indirect_input_ra) ? indirect_col_id + 1 : indirect_col_id,
        indirect_table_id,
        nest_level));
  }
}

template <class RA>
std::pair<std::vector<InputDescriptor>, std::list<std::shared_ptr<const InputColDescriptor>>> get_input_desc_impl(
    const RA* ra_node,
    const std::unordered_set<const RexInput*>& used_inputs,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::vector<InputDescriptor> input_descs;
  const auto data_sink_node = get_data_sink(ra_node);
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto input_ra = data_sink_node->getInput(nest_level);
    const int table_id = table_id_from_ra(input_ra);
    input_descs.emplace_back(table_id, nest_level);
  }
  std::unordered_set<std::shared_ptr<const InputColDescriptor>> input_col_descs_unique;
  collect_used_input_desc(input_descs, input_col_descs_unique, ra_node, used_inputs, input_to_nest_level);
  collect_used_input_desc(
      input_descs, input_col_descs_unique, ra_node, get_join_source_used_inputs(ra_node), input_to_nest_level);
  std::vector<std::shared_ptr<const InputColDescriptor>> input_col_descs(input_col_descs_unique.begin(),
                                                                         input_col_descs_unique.end());

  std::sort(
      input_col_descs.begin(),
      input_col_descs.end(),
      [](std::shared_ptr<const InputColDescriptor> const& lhs, std::shared_ptr<const InputColDescriptor> const& rhs) {
        if (lhs->getScanDesc().getNestLevel() == rhs->getScanDesc().getNestLevel()) {
          return lhs->getColId() < rhs->getColId();
        }
        return lhs->getScanDesc().getNestLevel() < rhs->getScanDesc().getNestLevel();
      });
  return {input_descs,
          std::list<std::shared_ptr<const InputColDescriptor>>(input_col_descs.begin(), input_col_descs.end())};
}

template <class RA>
std::tuple<std::vector<InputDescriptor>,
           std::list<std::shared_ptr<const InputColDescriptor>>,
           std::vector<std::shared_ptr<RexInput>>>
get_input_desc(const RA* ra_node, const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  std::tie(used_inputs, used_inputs_owned) = get_used_inputs(ra_node);
  auto input_desc_pair = get_input_desc_impl(ra_node, used_inputs, input_to_nest_level);
  return std::make_tuple(input_desc_pair.first, input_desc_pair.second, used_inputs_owned);
}

size_t get_scalar_sources_size(const RelCompound* compound) {
  return compound->getScalarSourcesSize();
}

size_t get_scalar_sources_size(const RelProject* project) {
  return project->size();
}

const RexScalar* scalar_at(const size_t i, const RelCompound* compound) {
  return compound->getScalarSource(i);
}

const RexScalar* scalar_at(const size_t i, const RelProject* project) {
  return project->getProjectAt(i);
}

std::shared_ptr<Analyzer::Expr> set_transient_dict(const std::shared_ptr<Analyzer::Expr> expr) {
  const auto& ti = expr->get_type_info();
  if (!ti.is_string() || ti.get_compression() != kENCODING_NONE) {
    return expr;
  }
  auto transient_dict_ti = ti;
  transient_dict_ti.set_compression(kENCODING_DICT);
  transient_dict_ti.set_comp_param(TRANSIENT_DICT_ID);
  transient_dict_ti.set_fixed_size();
  return expr->add_cast(transient_dict_ti);
}

template <class RA>
std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources(const RA* ra_node,
                                                                      const RelAlgTranslator& translator) {
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < get_scalar_sources_size(ra_node); ++i) {
    const auto scalar_rex = scalar_at(i, ra_node);
    if (dynamic_cast<const RexRef*>(scalar_rex)) {
      // RexRef are synthetic scalars we append at the end of the real ones
      // for the sake of taking memory ownership, no real work needed here.
      continue;
    }
    scalar_sources.push_back(translator.translateScalarRex(scalar_rex));
  }
  return scalar_sources;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelCompound* compound,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  if (!compound->isAggregate()) {
    return {nullptr};
  }
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (size_t group_idx = 0; group_idx < compound->getGroupByCount(); ++group_idx) {
    groupby_exprs.push_back(set_transient_dict(scalar_sources[group_idx]));
  }
  return groupby_exprs;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelAggregate* aggregate,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (size_t group_idx = 0; group_idx < aggregate->getGroupByCount(); ++group_idx) {
    groupby_exprs.push_back(set_transient_dict(scalar_sources[group_idx]));
  }
  return groupby_exprs;
}

struct QualsConjunctiveForm {
  const std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> quals;
};

QualsConjunctiveForm qual_to_conjunctive_form(const std::shared_ptr<Analyzer::Expr> qual_expr) {
  CHECK(qual_expr);
  const auto bin_oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(qual_expr);
  if (!bin_oper) {
    return {{}, {qual_expr}};
  }
  if (bin_oper->get_optype() == kAND) {
    const auto lhs_cf = qual_to_conjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_cf = qual_to_conjunctive_form(bin_oper->get_own_right_operand());
    auto simple_quals = lhs_cf.simple_quals;
    simple_quals.insert(simple_quals.end(), rhs_cf.simple_quals.begin(), rhs_cf.simple_quals.end());
    auto quals = lhs_cf.quals;
    quals.insert(quals.end(), rhs_cf.quals.begin(), rhs_cf.quals.end());
    return {simple_quals, quals};
  }
  int rte_idx{0};
  const auto simple_qual = bin_oper->normalize_simple_predicate(rte_idx);
  return simple_qual ? QualsConjunctiveForm{{simple_qual}, {}} : QualsConjunctiveForm{{}, {qual_expr}};
}

QualsConjunctiveForm translate_quals(const RelCompound* compound, const RelAlgTranslator& translator) {
  const auto filter_rex = compound->getFilterExpr();
  const auto filter_expr = filter_rex ? translator.translateScalarRex(filter_rex) : nullptr;
  return filter_expr ? qual_to_conjunctive_form(filter_expr) : QualsConjunctiveForm{};
}

std::vector<Analyzer::Expr*> translate_targets(std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
                                               const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
                                               const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
                                               const RelCompound* compound,
                                               const RelAlgTranslator& translator) {
  std::vector<Analyzer::Expr*> target_exprs;
  for (size_t i = 0; i < compound->size(); ++i) {
    const auto target_rex = compound->getTargetExpr(i);
    const auto target_rex_agg = dynamic_cast<const RexAgg*>(target_rex);
    std::shared_ptr<Analyzer::Expr> target_expr;
    if (target_rex_agg) {
      target_expr = RelAlgTranslator::translateAggregateRex(target_rex_agg, scalar_sources);
    } else {
      const auto target_rex_scalar = dynamic_cast<const RexScalar*>(target_rex);
      const auto target_rex_ref = dynamic_cast<const RexRef*>(target_rex_scalar);
      if (target_rex_ref) {
        const auto ref_idx = target_rex_ref->getIndex();
        CHECK_GE(ref_idx, size_t(1));
        CHECK_LE(ref_idx, groupby_exprs.size());
        const auto groupby_expr = *std::next(groupby_exprs.begin(), ref_idx - 1);
        target_expr = var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, ref_idx);
      } else {
        target_expr = translator.translateScalarRex(target_rex_scalar);
      }
    }
    CHECK(target_expr);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

std::vector<Analyzer::Expr*> translate_targets(std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
                                               const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
                                               const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
                                               const RelAggregate* aggregate,
                                               const RelAlgTranslator& translator) {
  std::vector<Analyzer::Expr*> target_exprs;
  size_t group_key_idx = 0;
  for (const auto& groupby_expr : groupby_exprs) {
    auto target_expr = var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, group_key_idx++);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }

  for (const auto& target_rex_agg : aggregate->getAggExprs()) {
    auto target_expr = RelAlgTranslator::translateAggregateRex(target_rex_agg.get(), scalar_sources);
    CHECK(target_expr);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

bool is_count_distinct(const Analyzer::Expr* expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(expr);
  return agg_expr && agg_expr->get_is_distinct();
}

template <class RA>
std::vector<TargetMetaInfo> get_targets_meta(const RA* ra_node, const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<TargetMetaInfo> targets_meta;
  for (size_t i = 0; i < ra_node->size(); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(
        ra_node->getFieldName(i),
        is_count_distinct(target_exprs[i]) ? SQLTypeInfo(kBIGINT, false) : target_exprs[i]->get_type_info());
  }
  return targets_meta;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeCompound(const RelCompound* compound,
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo,
                                                RenderInfo* render_info,
                                                const int64_t queue_time_ms) {
  const auto work_unit = createCompoundWorkUnit(compound, {{}, SortAlgorithm::Default, 0, 0});
  return executeWorkUnit(
      work_unit, compound->getOutputMetainfo(), compound->isAggregate(), co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeAggregate(const RelAggregate* aggregate,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo,
                                                 RenderInfo* render_info,
                                                 const int64_t queue_time_ms) {
  const auto work_unit = createAggregateWorkUnit(aggregate, {{}, SortAlgorithm::Default, 0, 0});
  return executeWorkUnit(work_unit, aggregate->getOutputMetainfo(), true, co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeProject(const RelProject* project,
                                               const CompilationOptions& co,
                                               const ExecutionOptions& eo,
                                               RenderInfo* render_info,
                                               const int64_t queue_time_ms) {
  const auto work_unit = createProjectWorkUnit(project, {{}, SortAlgorithm::Default, 0, 0});
  return executeWorkUnit(work_unit, project->getOutputMetainfo(), false, co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeFilter(const RelFilter* filter,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo,
                                              RenderInfo* render_info,
                                              const int64_t queue_time_ms) {
  const auto work_unit = createFilterWorkUnit(filter, {{}, SortAlgorithm::Default, 0, 0});
  return executeWorkUnit(work_unit, filter->getOutputMetainfo(), false, co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeJoin(const RelJoin* join,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            RenderInfo* render_info,
                                            const int64_t queue_time_ms) {
  const auto work_unit = createJoinWorkUnit(join, {{}, SortAlgorithm::Default, 0, 0});
  return executeWorkUnit(work_unit, join->getOutputMetainfo(), false, co, eo, render_info, queue_time_ms);
}

namespace {

// TODO(alex): Once we're fully migrated to the relational algebra model, change
// the executor interface to use the collation directly and remove this conversion.
std::list<Analyzer::OrderEntry> get_order_entries(const RelSort* sort) {
  std::list<Analyzer::OrderEntry> result;
  for (size_t i = 0; i < sort->collationCount(); ++i) {
    const auto sort_field = sort->getCollation(i);
    result.emplace_back(sort_field.getField() + 1,
                        sort_field.getSortDir() == SortDirection::Descending,
                        sort_field.getNullsPosition() == NullSortedPosition::First);
  }
  return result;
}

size_t get_scan_limit(const RelAlgNode* ra, const size_t limit) {
  const auto aggregate = dynamic_cast<const RelAggregate*>(ra);
  if (aggregate) {
    return 0;
  }
  const auto compound = dynamic_cast<const RelCompound*>(ra);
  return (compound && compound->isAggregate()) ? 0 : limit;
}

bool first_oe_is_desc(const std::list<Analyzer::OrderEntry>& order_entries) {
  return !order_entries.empty() && order_entries.front().is_desc;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeSort(const RelSort* sort,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            RenderInfo* render_info,
                                            const int64_t queue_time_ms) {
  CHECK_EQ(size_t(1), sort->inputCount());
  const auto source = sort->getInput(0);
  if (dynamic_cast<const RelSort*>(source)) {
    throw std::runtime_error("Sort node not supported as input to another sort");
  }
  const auto compound = dynamic_cast<const RelCompound*>(source);
  const auto aggregate = dynamic_cast<const RelAggregate*>(source);
  const bool is_aggregate = ((compound && compound->isAggregate()) || aggregate);
  while (true) {
    std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
    bool is_desc{false};
    try {
      const auto source_work_unit = createSortInputWorkUnit(sort);
      is_desc = first_oe_is_desc(source_work_unit.exe_unit.sort_info.order_entries);
      groupby_exprs = source_work_unit.exe_unit.groupby_exprs;
      auto source_result = executeWorkUnit(
          source_work_unit, source->getOutputMetainfo(), is_aggregate, co, eo, render_info, queue_time_ms);
      if (render_info && render_info->do_render) {
        return source_result;
      }
      auto rows_to_sort = source_result.getRows();
      if (eo.just_explain) {
        return {rows_to_sort, {}};
      }
      const size_t limit = sort->getLimit();
      const size_t offset = sort->getOffset();
      if (sort->collationCount() != 0 &&
          !use_speculative_top_n(source_work_unit.exe_unit, rows_to_sort.getQueryMemDesc())) {
        rows_to_sort.sort(source_work_unit.exe_unit.sort_info.order_entries, false, limit + offset);
      }
      if (limit || offset) {
        rows_to_sort.dropFirstN(offset);
        if (limit) {
          rows_to_sort.keepFirstN(limit);
        }
      }
      return {rows_to_sort, source_result.getTargetsMeta()};
    } catch (const SpeculativeTopNFailed&) {
      CHECK_EQ(size_t(1), groupby_exprs.size());
      speculative_topn_blacklist_.add(groupby_exprs.front(), is_desc);
    }
  }
  CHECK(false);
  return {ResultRows({}, {}, nullptr, nullptr, {}, co.device_type_), {}};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createSortInputWorkUnit(const RelSort* sort) {
  const auto source = sort->getInput(0);
  const size_t limit = sort->getLimit();
  const size_t offset = sort->getOffset();
  const size_t scan_limit = sort->collationCount() ? 0 : get_scan_limit(source, limit);
  const size_t scan_total_limit = scan_limit ? get_scan_limit(source, scan_limit + offset) : 0;
  size_t max_groups_buffer_entry_guess{scan_total_limit ? scan_total_limit : max_groups_buffer_entry_default_guess};
  SortAlgorithm sort_algorithm{SortAlgorithm::SpeculativeTopN};
  const auto order_entries = get_order_entries(sort);
  SortInfo sort_info{order_entries, sort_algorithm, limit, offset};
  auto source_work_unit = createWorkUnit(source, sort_info);
  const auto& source_exe_unit = source_work_unit.exe_unit;
  if (source_exe_unit.groupby_exprs.size() == 1 && source_exe_unit.groupby_exprs.front() &&
      speculative_topn_blacklist_.contains(source_exe_unit.groupby_exprs.front(), first_oe_is_desc(order_entries))) {
    sort_algorithm = SortAlgorithm::Default;
  }
  sort->setOutputMetainfo(source->getOutputMetainfo());
  return {{source_exe_unit.input_descs,
           source_exe_unit.extra_input_descs,
           std::move(source_exe_unit.input_col_descs),
           source_exe_unit.simple_quals,
           source_exe_unit.quals,
           source_exe_unit.join_type,
           source_exe_unit.join_dimensions,
           source_exe_unit.inner_join_quals,
           source_exe_unit.outer_join_quals,
           source_exe_unit.groupby_exprs,
           source_exe_unit.target_exprs,
           source_exe_unit.orig_target_exprs,
           nullptr,
           {sort_info.order_entries, sort_algorithm, limit, offset},
           scan_total_limit},
          max_groups_buffer_entry_guess,
          std::move(source_work_unit.query_rewriter)};
}

namespace {

// Upper bound estimation for the number of groups. Not strictly correct and not
// tight, but if the tables involved are really small we shouldn't waste time doing
// the NDV estimation. We don't account for cross-joins and / or group by unnested
// array, which is the reason this estimation isn't entirely reliable.
size_t groups_approx_upper_bound(const std::vector<InputTableInfo>& table_infos) {
  CHECK(!table_infos.empty());
  const auto& first_table = table_infos.front();
  size_t max_num_groups = first_table.info.numTuples;
  for (const auto& table_info : table_infos) {
    if (table_info.info.numTuples > max_num_groups) {
      max_num_groups = table_info.info.numTuples;
    }
  }
  return std::max(max_num_groups, size_t(1));
}

}  // namespace

ExecutionResult RelAlgExecutor::executeWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                                const std::vector<TargetMetaInfo>& targets_meta,
                                                const bool is_agg,
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo,
                                                RenderInfo* render_info,
                                                const int64_t queue_time_ms) {
  int32_t error_code{0};
  if (render_info && render_info->do_render) {
    if (co.device_type_ != ExecutorDeviceType::GPU) {
      throw std::runtime_error("Backend rendering is only supported on GPU");
    }
    if (!executor_->render_manager_) {
      throw std::runtime_error("This build doesn't support backend rendering");
    }

    if (!render_info->render_allocator_map_ptr) {
      // for backwards compatibility, can be removed when MapDHandler::render(...)
      // in MapDServer.cpp is removed
      render_info->render_allocator_map_ptr.reset(new RenderAllocatorMap(
          cat_.get_dataMgr().cudaMgr_, executor_->render_manager_, executor_->blockSize(), executor_->gridSize()));
    }
  }

  const auto table_infos = get_table_infos(work_unit.exe_unit, cat_, temporary_tables_);

  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;

  static const size_t big_group_threshold{20000};

  ExecutionResult result{ResultRows({}, {}, nullptr, nullptr, {}, co.device_type_), {}};

  try {
    result = {executor_->executeWorkUnit(
                  &error_code,
                  max_groups_buffer_entry_guess,
                  is_agg,
                  table_infos,
                  work_unit.exe_unit,
                  co,
                  eo,
                  cat_,
                  executor_->row_set_mem_owner_,
                  (render_info && render_info->do_render ? render_info->render_allocator_map_ptr.get() : nullptr),
                  groups_approx_upper_bound(table_infos) <= big_group_threshold),
              targets_meta};
  } catch (const CardinalityEstimationRequired&) {
    max_groups_buffer_entry_guess =
        2 * std::min(groups_approx_upper_bound(table_infos), getNDVEstimation(work_unit, targets_meta, is_agg, co, eo));
    CHECK_GT(max_groups_buffer_entry_guess, size_t(0));
    result = {executor_->executeWorkUnit(
                  &error_code,
                  max_groups_buffer_entry_guess,
                  is_agg,
                  table_infos,
                  work_unit.exe_unit,
                  co,
                  eo,
                  cat_,
                  executor_->row_set_mem_owner_,
                  (render_info && render_info->do_render ? render_info->render_allocator_map_ptr.get() : nullptr),
                  true),
              targets_meta};
  }
  result.setQueueTime(queue_time_ms);
#ifdef HAVE_RENDERING
  if (render_info && render_info->do_render) {
    return renderWorkUnit(work_unit, targets_meta, render_info, error_code, queue_time_ms);
  }
#endif  // HAVE_RENDERING
  if (!error_code) {
    return result;
  }
  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW) {
    throw std::runtime_error("Overflow or underflow");
  }
  if (error_code == Executor::ERR_DIV_BY_ZERO) {
    throw std::runtime_error("Division by zero");
  }
  if (error_code == Executor::ERR_UNSUPPORTED_SELF_JOIN) {
    throw std::runtime_error("Self joins not supported yet");
  }
  if (error_code == Executor::ERR_OUT_OF_CPU_MEM) {
    throw std::runtime_error("Not enough host memory to execute the query");
  }
  if (error_code == Executor::ERR_SPECULATIVE_TOP_OOM) {
    throw SpeculativeTopNFailed();
  }
  return handleRetry(
      error_code, {work_unit.exe_unit, max_groups_buffer_entry_guess}, targets_meta, is_agg, co, eo, queue_time_ms);
}

size_t RelAlgExecutor::getNDVEstimation(const WorkUnit& work_unit,
                                        const std::vector<TargetMetaInfo>& targets_meta,
                                        const bool is_agg,
                                        const CompilationOptions& co,
                                        const ExecutionOptions& eo) {
  int32_t error_code{0};
  const auto estimator_exe_unit = create_ndv_execution_unit(work_unit.exe_unit);
  size_t one{1};
  const auto estimator_result = executor_->executeWorkUnit(&error_code,
                                                           one,
                                                           is_agg,
                                                           get_table_infos(work_unit.exe_unit, cat_, temporary_tables_),
                                                           estimator_exe_unit,
                                                           co,
                                                           eo,
                                                           cat_,
                                                           executor_->row_set_mem_owner_,
                                                           nullptr,
                                                           false);
  if (error_code) {
    return std::numeric_limits<size_t>::max();
  }
  const auto& estimator_result_rows = boost::get<RowSetPtr>(estimator_result);
  CHECK(estimator_result_rows);
  const auto estimator_result_set = estimator_result_rows->getResultSet();
  CHECK(estimator_result_set);
  return std::max(estimator_result_set->getNDVEstimator(), size_t(1));
}

#ifdef HAVE_RENDERING
ExecutionResult RelAlgExecutor::renderWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                               const std::vector<TargetMetaInfo>& targets_meta,
                                               RenderInfo* render_info,
                                               const int32_t error_code,
                                               const int64_t queue_time_ms) {
  if (error_code == Executor::ERR_OUT_OF_RENDER_MEM) {
    throw std::runtime_error("Not enough OpenGL memory to render the query results");
  }
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM) {
    throw std::runtime_error("Not enough GPU memory to execute the query");
  }
  if (error_code && !work_unit.exe_unit.scan_limit) {
    CHECK_LT(error_code, 0);
    throw std::runtime_error("Ran out of slots in the output buffer");
  }
  auto clock_begin = timer_start();
  CHECK_EQ(target_exprs_owned_.size(), targets_meta.size());
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> target_entries;
  for (size_t i = 0; i < targets_meta.size(); ++i) {
    target_entries.emplace_back(
        new Analyzer::TargetEntry(targets_meta[i].get_resname(), target_exprs_owned_[i], false));
  }
  auto image_bytes = executor_->renderRows(target_entries, render_info);
  int64_t render_time_ms = timer_stop(clock_begin);
  return {boost::make_unique<ResultRows>(image_bytes, queue_time_ms, render_time_ms), {}};
}
#endif  // HAVE_RENDERING

ExecutionResult RelAlgExecutor::handleRetry(const int32_t error_code_in,
                                            const RelAlgExecutor::WorkUnit& work_unit,
                                            const std::vector<TargetMetaInfo>& targets_meta,
                                            const bool is_agg,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            const int64_t queue_time_ms) {
  auto error_code = error_code_in;
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  ExecutionOptions eo_no_multifrag{eo.output_columnar_hint, false, false, eo.allow_loop_joins, eo.with_watchdog};
  ExecutionResult result{ResultRows({}, {}, nullptr, nullptr, {}, co.device_type_), {}};
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM) {
    result = {executor_->executeWorkUnit(&error_code,
                                         max_groups_buffer_entry_guess,
                                         is_agg,
                                         get_table_infos(work_unit.exe_unit, cat_, temporary_tables_),
                                         work_unit.exe_unit,
                                         co,
                                         eo_no_multifrag,
                                         cat_,
                                         executor_->row_set_mem_owner_,
                                         nullptr,
                                         true),
              targets_meta};
    result.setQueueTime(queue_time_ms);
  }
  if (co.device_type_ == ExecutorDeviceType::GPU) {
    LOG(INFO) << "Query ran out of GPU memory, punt to CPU";
  }
  CompilationOptions co_cpu{ExecutorDeviceType::CPU, co.hoist_literals_, co.opt_level_};
  if (error_code) {
    max_groups_buffer_entry_guess = 0;
    while (true) {
      result = {executor_->executeWorkUnit(&error_code,
                                           max_groups_buffer_entry_guess,
                                           is_agg,
                                           get_table_infos(work_unit.exe_unit, cat_, temporary_tables_),
                                           work_unit.exe_unit,
                                           co_cpu,
                                           eo_no_multifrag,
                                           cat_,
                                           executor_->row_set_mem_owner_,
                                           nullptr,
                                           true),
                targets_meta};
      result.setQueueTime(queue_time_ms);
      if (!error_code) {
        return result;
      }
      if (error_code == Executor::ERR_OUT_OF_CPU_MEM) {
        throw std::runtime_error("Not enough host memory to execute the query");
      }
      // Even the conservative guess failed; it should only happen when we group
      // by a huge cardinality array. Maybe we should throw an exception instead?
      // Such a heavy query is entirely capable of exhausting all the host memory.
      CHECK(max_groups_buffer_entry_guess);
      max_groups_buffer_entry_guess *= 2;
    }
  }
  return result;
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createWorkUnit(const RelAlgNode* node, const SortInfo& sort_info) {
  const auto compound = dynamic_cast<const RelCompound*>(node);
  if (compound) {
    return createCompoundWorkUnit(compound, sort_info);
  }
  const auto project = dynamic_cast<const RelProject*>(node);
  if (project) {
    return createProjectWorkUnit(project, sort_info);
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(node);
  if (aggregate) {
    return createAggregateWorkUnit(aggregate, sort_info);
  }
  const auto filter = dynamic_cast<const RelFilter*>(node);
  CHECK(filter);
  return createFilterWorkUnit(filter, sort_info);
}

namespace {

class UsedTablesVisitor : public ScalarExprVisitor<std::unordered_set<int>> {
 protected:
  virtual std::unordered_set<int> visitColumnVar(const Analyzer::ColumnVar* column) const override {
    return {column->get_table_id()};
  }

  virtual std::unordered_set<int> aggregateResult(const std::unordered_set<int>& aggregate,
                                                  const std::unordered_set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

struct SeparatedQuals {
  const std::list<std::shared_ptr<Analyzer::Expr>> regular_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
};

SeparatedQuals separate_join_quals(const std::list<std::shared_ptr<Analyzer::Expr>>& all_quals) {
  std::list<std::shared_ptr<Analyzer::Expr>> regular_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
  UsedTablesVisitor qual_visitor;
  for (auto qual_candidate : all_quals) {
    const auto used_table_ids = qual_visitor.visit(qual_candidate.get());
    if (used_table_ids.size() > 1) {
      CHECK_EQ(size_t(2), used_table_ids.size());
      join_quals.push_back(qual_candidate);
    } else {
      regular_quals.push_back(qual_candidate);
    }
  }
  return {regular_quals, join_quals};
}

JoinType get_join_type(const RelAlgNode* ra) {
  const auto join_ra = dynamic_cast<const RelJoin*>(get_data_sink(ra));
  return join_ra ? join_ra->getJoinType() : JoinType::INVALID;
}

bool is_literal_true(const RexScalar* condition) {
  CHECK(condition);
  const auto literal = dynamic_cast<const RexLiteral*>(condition);
  return literal && literal->getType() == kBOOLEAN && literal->getVal<bool>();
}

std::list<std::shared_ptr<Analyzer::Expr>> get_outer_join_quals(const RelAlgNode* ra,
                                                                const RelAlgTranslator& translator) {
  const auto join = dynamic_cast<const RelJoin*>(ra) ? static_cast<const RelJoin*>(ra)
                                                     : dynamic_cast<const RelJoin*>(ra->getInput(0));
  if (join && join->getCondition() && !is_literal_true(join->getCondition()) && join->getJoinType() == JoinType::LEFT) {
    const auto join_cond_cf = qual_to_conjunctive_form(translator.translateScalarRex(join->getCondition()));
    CHECK(join_cond_cf.simple_quals.empty());
    return join_cond_cf.quals;
  }
  return {};
}

std::list<std::shared_ptr<Analyzer::Expr>> get_inner_join_quals(const RelAlgNode* ra,
                                                                const RelAlgTranslator& translator) {
  const auto join = dynamic_cast<const RelJoin*>(ra) ? static_cast<const RelJoin*>(ra)
                                                     : dynamic_cast<const RelJoin*>(ra->getInput(0));
  if (join && join->getCondition() && !is_literal_true(join->getCondition()) &&
      join->getJoinType() == JoinType::INNER) {
    const auto join_cond_cf = qual_to_conjunctive_form(translator.translateScalarRex(join->getCondition()));
    CHECK(join_cond_cf.simple_quals.empty());
    return join_cond_cf.quals;
  }
  return {};
}

std::vector<std::pair<int, size_t>> get_join_dimensions(const RelAlgNode* ra,
                                                        const Catalog_Namespace::Catalog& cat,
                                                        const TemporaryTables& temp_tables) {
  std::vector<std::pair<int, size_t>> dims;
  for (auto join = dynamic_cast<const RelJoin*>(ra); join; join = static_cast<const RelJoin*>(join->getInput(0))) {
    CHECK_EQ(size_t(2), join->inputCount());
    const auto id = table_id_from_ra(join->getInput(1));
    dims.emplace_back(id, get_frag_count_of_table(id, cat, temp_tables));
    auto lhs = join->getInput(0);
    if (!dynamic_cast<const RelJoin*>(lhs)) {
      const auto id = table_id_from_ra(lhs);
      dims.emplace_back(id, get_frag_count_of_table(id, cat, temp_tables));
      break;
    }
  }
  std::reverse(dims.begin(), dims.end());
  return dims;
}

std::vector<InputDescriptor> separate_extra_input_descs(std::vector<InputDescriptor>& input_descs) {
  std::vector<InputDescriptor> new_input_descs;
  std::vector<InputDescriptor> extra_input_descs;

  for (const auto& input_desc : input_descs) {
    if (input_desc.getNestLevel() < 0) {
      extra_input_descs.push_back(input_desc);
    } else {
      new_input_descs.push_back(input_desc);
    }
  }

  input_descs.swap(new_input_descs);
  return extra_input_descs;
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createCompoundWorkUnit(const RelCompound* compound,
                                                                const SortInfo& sort_info) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  const auto input_to_nest_level = get_input_nest_levels(compound);
  std::tie(input_descs, input_col_descs, std::ignore) = get_input_desc(compound, input_to_nest_level);
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = get_join_type(compound);
  RelAlgTranslator translator(cat_, input_to_nest_level, join_type, now_);
  const auto scalar_sources = translate_scalar_sources(compound, translator);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals_cf = translate_quals(compound, translator);
  const auto separated_quals = separate_join_quals(quals_cf.quals);
  const auto simple_separated_quals = separate_join_quals(quals_cf.simple_quals);
  CHECK(simple_separated_quals.join_quals.empty());
  const auto target_exprs = translate_targets(target_exprs_owned_, scalar_sources, groupby_exprs, compound, translator);
  CHECK_EQ(compound->size(), target_exprs.size());
  auto inner_join_quals = get_inner_join_quals(compound, translator);
  inner_join_quals.insert(inner_join_quals.end(), separated_quals.join_quals.begin(), separated_quals.join_quals.end());
  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        extra_input_descs,
                                        input_col_descs,
                                        quals_cf.simple_quals,
                                        separated_quals.regular_quals,
                                        join_type,
                                        get_join_dimensions(get_data_sink(compound), cat_, temporary_tables_),
                                        inner_join_quals,
                                        get_outer_join_quals(compound, translator),
                                        groupby_exprs,
                                        target_exprs,
                                        {},
                                        nullptr,
                                        sort_info,
                                        0};
  const auto query_infos = get_table_infos(exe_unit.input_descs, cat_, temporary_tables_);
  QueryRewriter* query_rewriter = new QueryRewriter(exe_unit, query_infos, executor_, nullptr);
  const auto rewritten_exe_unit = query_rewriter->rewrite();
  const auto targets_meta = get_targets_meta(compound, rewritten_exe_unit.target_exprs);
  compound->setOutputMetainfo(targets_meta);
  return {rewritten_exe_unit, max_groups_buffer_entry_default_guess, std::unique_ptr<QueryRewriter>(query_rewriter)};
}

namespace {

std::vector<TargetMetaInfo> get_inputs_meta(const RelScan* scan, const Catalog_Namespace::Catalog& cat) {
  std::vector<TargetMetaInfo> in_metainfo;
  for (const auto& col_name : scan->getFieldNames()) {
    const auto table_desc = scan->getTableDescriptor();
    const auto cd = cat.getMetadataForColumn(table_desc->tableId, col_name);
    CHECK(cd);
    auto col_ti = cd->columnType;
    in_metainfo.emplace_back(col_name, col_ti);
  }
  return in_metainfo;
}

std::vector<std::shared_ptr<Analyzer::Expr>> get_input_exprs(const RelJoin* join, const bool need_original) {
  const auto join_type = join->getJoinType();
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;
  const auto lhs = join->getInput(0);
  if (need_original && dynamic_cast<const RelJoin*>(lhs)) {
    const auto previous_join = static_cast<const RelJoin*>(lhs);
    auto source_exprs_owned = get_input_exprs(previous_join, true);
    for (size_t i = 0; i < source_exprs_owned.size(); ++i) {
      const auto iter_ti = source_exprs_owned[i]->get_type_info();
      auto iter_expr = std::make_shared<Analyzer::IterExpr>(iter_ti, table_id_from_ra(lhs), 0);
      target_exprs_owned.push_back(iter_expr);
    }
  } else {
    const auto iter_ti = SQLTypeInfo(kBIGINT, true);
    auto iter_expr = std::make_shared<Analyzer::IterExpr>(iter_ti, table_id_from_ra(lhs), 0);
    target_exprs_owned.push_back(iter_expr);
  }

  const auto rhs = join->getInput(1);
  CHECK(!dynamic_cast<const RelJoin*>(rhs));
  const auto iter_ti = SQLTypeInfo(kBIGINT, join_type == JoinType::INNER);
  auto iter_expr = std::make_shared<Analyzer::IterExpr>(iter_ti, table_id_from_ra(rhs), 1);
  target_exprs_owned.push_back(iter_expr);

  return target_exprs_owned;
}

std::pair<std::vector<TargetMetaInfo>, std::vector<std::shared_ptr<Analyzer::Expr>>> get_inputs_meta(
    const RelJoin* join,
    const Catalog_Namespace::Catalog& cat) {
  std::vector<TargetMetaInfo> targets_meta;
  const auto lhs = join->getInput(0);
  if (auto scan = dynamic_cast<const RelScan*>(lhs)) {
    const auto lhs_in_meta = get_inputs_meta(scan, cat);
    targets_meta.insert(targets_meta.end(), lhs_in_meta.begin(), lhs_in_meta.end());
  } else {
    const auto& lhs_in_meta = lhs->getOutputMetainfo();
    targets_meta.insert(targets_meta.end(), lhs_in_meta.begin(), lhs_in_meta.end());
  }
  const auto rhs = join->getInput(1);
  CHECK(!dynamic_cast<const RelJoin*>(rhs));
  if (auto scan = dynamic_cast<const RelScan*>(rhs)) {
    const auto rhs_in_meta = get_inputs_meta(scan, cat);
    targets_meta.insert(targets_meta.end(), rhs_in_meta.begin(), rhs_in_meta.end());
  } else {
    const auto& rhs_in_meta = rhs->getOutputMetainfo();
    targets_meta.insert(targets_meta.end(), rhs_in_meta.begin(), rhs_in_meta.end());
  }
  return std::make_pair(targets_meta, get_input_exprs(join, false));
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createJoinWorkUnit(const RelJoin* join, const SortInfo& sort_info) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  const auto input_to_nest_level = get_input_nest_levels(join);
  std::tie(input_descs, input_col_descs, std::ignore) = get_input_desc(join, input_to_nest_level);
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = join->getJoinType();
  RelAlgTranslator translator(cat_, input_to_nest_level, join_type, now_);
  auto inner_join_quals = get_inner_join_quals(join, translator);
  auto outer_join_quals = get_outer_join_quals(join, translator);
  CHECK((join_type == JoinType::INNER && outer_join_quals.empty()) ||
        (join_type == JoinType::LEFT && inner_join_quals.empty()));
  std::vector<TargetMetaInfo> targets_meta;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;
  std::tie(targets_meta, target_exprs_owned) = get_inputs_meta(join, cat_);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  auto orig_target_exprs_owned = get_input_exprs(join, true);
  target_exprs_owned_.insert(target_exprs_owned_.end(), orig_target_exprs_owned.begin(), orig_target_exprs_owned.end());
  join->setOutputMetainfo(targets_meta);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           {},
           join_type,
           get_join_dimensions(join, cat_, temporary_tables_),
           inner_join_quals,
           outer_join_quals,
           {nullptr},
           get_exprs_not_owned(target_exprs_owned),
           get_exprs_not_owned(orig_target_exprs_owned),
           nullptr,
           sort_info,
           0},
          max_groups_buffer_entry_default_guess,
          nullptr};
}

namespace {

std::vector<std::shared_ptr<Analyzer::Expr>> synthesize_inputs(
    const RelAlgNode* ra_node,
    const size_t nest_level,
    const std::vector<TargetMetaInfo>& in_metainfo,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  CHECK_LE(size_t(1), ra_node->inputCount());
  CHECK_GE(size_t(2), ra_node->inputCount());
  const auto input = ra_node->getInput(nest_level);
  const auto it_rte_idx = input_to_nest_level.find(input);
  CHECK(it_rte_idx != input_to_nest_level.end());
  const int rte_idx = it_rte_idx->second;
  const int table_id = table_id_from_ra(input);
  std::vector<std::shared_ptr<Analyzer::Expr>> inputs;
  const auto scan_ra = dynamic_cast<const RelScan*>(input);
  int input_idx = 0;
  for (const auto& input_meta : in_metainfo) {
    inputs.push_back(std::make_shared<Analyzer::ColumnVar>(
        input_meta.get_type_info(), table_id, scan_ra ? input_idx + 1 : input_idx, rte_idx));
    ++input_idx;
  }
  return inputs;
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createAggregateWorkUnit(const RelAggregate* aggregate,
                                                                 const SortInfo& sort_info) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto input_to_nest_level = get_input_nest_levels(aggregate);
  std::tie(input_descs, input_col_descs, used_inputs_owned) = get_input_desc(aggregate, input_to_nest_level);
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = get_join_type(aggregate);
  RelAlgTranslator translator(cat_, input_to_nest_level, join_type, now_);
  CHECK_EQ(size_t(1), aggregate->inputCount());
  const auto source = aggregate->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  const auto scalar_sources = synthesize_inputs(aggregate, size_t(0), in_metainfo, input_to_nest_level);
  const auto groupby_exprs = translate_groupby_exprs(aggregate, scalar_sources);
  const auto target_exprs =
      translate_targets(target_exprs_owned_, scalar_sources, groupby_exprs, aggregate, translator);
  const auto targets_meta = get_targets_meta(aggregate, target_exprs);
  aggregate->setOutputMetainfo(targets_meta);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           {},
           join_type,
           get_join_dimensions(get_data_sink(aggregate), cat_, temporary_tables_),
           get_inner_join_quals(aggregate, translator),
           get_outer_join_quals(aggregate, translator),
           groupby_exprs,
           target_exprs,
           {},
           nullptr,
           sort_info,
           0},
          max_groups_buffer_entry_default_guess,
          nullptr};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createProjectWorkUnit(const RelProject* project, const SortInfo& sort_info) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  const auto input_to_nest_level = get_input_nest_levels(project);
  std::tie(input_descs, input_col_descs, std::ignore) = get_input_desc(project, input_to_nest_level);
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = get_join_type(project);
  RelAlgTranslator translator(cat_, input_to_nest_level, join_type, now_);
  const auto target_exprs_owned = translate_scalar_sources(project, translator);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  const auto targets_meta = get_targets_meta(project, target_exprs);
  project->setOutputMetainfo(targets_meta);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           {},
           join_type,
           get_join_dimensions(get_data_sink(project), cat_, temporary_tables_),
           get_inner_join_quals(project, translator),
           get_outer_join_quals(project, translator),
           {nullptr},
           target_exprs,
           {},
           nullptr,
           sort_info,
           0},
          max_groups_buffer_entry_default_guess,
          nullptr};
}

namespace {

std::pair<std::vector<TargetMetaInfo>, std::vector<std::shared_ptr<Analyzer::Expr>>> get_inputs_meta(
    const RelFilter* filter,
    const RelAlgTranslator& translator,
    const std::vector<std::shared_ptr<RexInput>>& inputs_owned,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::vector<TargetMetaInfo> in_metainfo;
  std::vector<std::shared_ptr<Analyzer::Expr>> exprs_owned;
  const auto data_sink_node = get_data_sink(filter);
  auto input_it = inputs_owned.begin();
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto source = data_sink_node->getInput(nest_level);
    const auto scan_source = dynamic_cast<const RelScan*>(source);
    if (scan_source) {
      CHECK(source->getOutputMetainfo().empty());
      std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources_owned;
      for (size_t i = 0; i < scan_source->size(); ++i, ++input_it) {
        scalar_sources_owned.push_back(translator.translateScalarRex(input_it->get()));
      }
      const auto source_metadata = get_targets_meta(scan_source, get_exprs_not_owned(scalar_sources_owned));
      in_metainfo.insert(in_metainfo.end(), source_metadata.begin(), source_metadata.end());
      exprs_owned.insert(exprs_owned.end(), scalar_sources_owned.begin(), scalar_sources_owned.end());
    } else {
      const auto& source_metadata = source->getOutputMetainfo();
      input_it += source_metadata.size();
      in_metainfo.insert(in_metainfo.end(), source_metadata.begin(), source_metadata.end());
      const auto scalar_sources_owned =
          synthesize_inputs(data_sink_node, nest_level, source_metadata, input_to_nest_level);
      exprs_owned.insert(exprs_owned.end(), scalar_sources_owned.begin(), scalar_sources_owned.end());
    }
  }
  return std::make_pair(in_metainfo, exprs_owned);
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createFilterWorkUnit(const RelFilter* filter, const SortInfo& sort_info) {
  CHECK_EQ(size_t(1), filter->inputCount());
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<TargetMetaInfo> in_metainfo;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;

  const auto input_to_nest_level = get_input_nest_levels(filter);
  std::tie(input_descs, input_col_descs, used_inputs_owned) = get_input_desc(filter, input_to_nest_level);
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = get_join_type(filter);
  RelAlgTranslator translator(cat_, input_to_nest_level, join_type, now_);
  std::tie(in_metainfo, target_exprs_owned) =
      get_inputs_meta(filter, translator, used_inputs_owned, input_to_nest_level);
  const auto qual = translator.translateScalarRex(filter->getCondition());
  std::list<std::shared_ptr<Analyzer::Expr>> quals{qual};
  const auto separated_quals = separate_join_quals(quals);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  filter->setOutputMetainfo(in_metainfo);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           separated_quals.regular_quals,
           join_type,
           get_join_dimensions(get_data_sink(filter), cat_, temporary_tables_),
           separated_quals.join_quals,
           get_outer_join_quals(filter, translator),
           {nullptr},
           target_exprs,
           {},
           nullptr,
           sort_info,
           0},
          max_groups_buffer_entry_default_guess,
          nullptr};
}

SpeculativeTopNBlacklist RelAlgExecutor::speculative_topn_blacklist_;

#endif  // HAVE_CALCITE
