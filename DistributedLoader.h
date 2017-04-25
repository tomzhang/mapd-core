#ifndef DISTRIBUTEDLOADER_H
#define DISTRIBUTEDLOADER_H

#include "Import/Importer.h"
#include "LeafAggregator.h"

#include <atomic>
#include <mutex>
#include <unordered_set>

class DistributedLoader : public Importer_NS::Loader {
 public:
  DistributedLoader(const Catalog_Namespace::SessionInfo& parent_session_info,
                    const TableDescriptor* t,
                    LeafAggregator* aggregator);

  bool load(const std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>& import_buffers,
            size_t row_count) override;
  bool loadNoCheckpoint(const std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>& import_buffers,
                        size_t row_count) override;

 private:
  bool loadDistImpl(const std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>& import_buffers,
                    size_t row_count);
  const Catalog_Namespace::SessionInfo& parent_session_info_;
  LeafAggregator* aggregator_;
  std::atomic<uint64_t> load_call_count_;
  std::unordered_set<size_t> leaves_pending_checkpoint_;
  std::mutex leaves_pending_checkpoint_mutex_;
};

#endif  // DISTRIBUTEDLOADER_H
