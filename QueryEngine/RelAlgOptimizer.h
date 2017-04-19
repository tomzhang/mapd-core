#ifndef QUERYENGINE_RELALGOPTIMIZER_H
#define QUERYENGINE_RELALGOPTIMIZER_H

#include <memory>
#include <vector>

class RelAlgNode;

void eliminate_identical_copy(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void eliminate_dead_columns(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void fold_filters(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void hoist_filter_cond_to_cross_join(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void simplify_sort(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;

#endif  // QUERYENGINE_RELALGOPTIMIZER_H
