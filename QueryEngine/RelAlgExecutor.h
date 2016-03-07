#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "Execute.h"
#include "RelAlgExecutionDescriptor.h"

class RelAlgExecutor {
 public:
  RelAlgExecutor(Executor* executor, const Catalog_Namespace::Catalog& cat) : executor_(executor), cat_(cat) {}
  ExecutionResult executeRelAlgSeq(std::vector<RaExecutionDesc>&, const CompilationOptions&);

 private:
  ExecutionResult executeCompound(const RelCompound*, const CompilationOptions&);
  ExecutionResult executeProject(const RelProject*, const CompilationOptions&);

  ExecutionResult executeWorkUnit(const Executor::RelAlgExecutionUnit& rel_alg_exe_unit,
                                  const std::vector<ScanDescriptor>& scan_ids,
                                  const std::vector<Analyzer::TargetMetaInfo>& targets_meta,
                                  const bool is_agg,
                                  const CompilationOptions& co);

  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
};

#endif  // QUERYENGINE_RELALGEXECUTOR_H
