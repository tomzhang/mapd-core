#ifndef QUERYENGINE_QUERYTEMPLATEGENERATOR_H
#define QUERYENGINE_QUERYTEMPLATEGENERATOR_H

#include "GroupByAndAggregate.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <string>

std::string unique_name(const char* base_name, const bool is_nested);
llvm::Function* query_template(llvm::Module*, const size_t aggr_col_count, const bool is_nested, const bool hoist_literals);
llvm::Function* query_group_by_template(llvm::Module*, const size_t aggr_col_count, const bool is_nested,
                                        const bool hoist_literals, const QueryMemoryDescriptor& query_mem_desc);

#endif  // QUERYENGINE_QUERYTEMPLATEGENERATOR_H
