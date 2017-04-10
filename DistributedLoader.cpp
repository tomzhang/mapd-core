#include "DistributedLoader.h"

DistributedLoader::DistributedLoader(const Catalog_Namespace::SessionInfo& parent_session_info,
                                     const TableDescriptor* t,
                                     LeafAggregator* aggregator)
    : Loader(parent_session_info.get_catalog(), t),
      parent_session_info_(parent_session_info),
      aggregator_(aggregator),
      load_call_count_(0) {}

bool DistributedLoader::load(const std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>& import_buffers,
                             size_t row_count) {
  TInsertData thrift_insert_data;
  thrift_insert_data.db_id = insert_data.databaseId;
  thrift_insert_data.table_id = insert_data.tableId;
  thrift_insert_data.column_ids = insert_data.columnIds;
  thrift_insert_data.num_rows = row_count;
  for (const auto& import_buff : import_buffers) {
    TDataBlockPtr p;
    if (import_buff->getTypeInfo().is_number() || import_buff->getTypeInfo().is_time() ||
        import_buff->getTypeInfo().get_type() == kBOOLEAN) {
      const auto ptr = reinterpret_cast<const char*>(import_buff->getAsBytes());
      const auto size = import_buff->getElementSize() * row_count;
      p.fixed_len_data = std::string(ptr, size);
    } else if (import_buff->getTypeInfo().is_string()) {
      auto string_payload_ptr = import_buff->getStringBuffer();
      CHECK_EQ(row_count, string_payload_ptr->size());
      if (import_buff->getTypeInfo().get_compression() == kENCODING_NONE) {
        for (const auto& str : *string_payload_ptr) {
          TVarLen varlen_str;
          varlen_str.payload = str;
          varlen_str.is_null = false;
          p.var_len_data.push_back(varlen_str);
        }
      } else {
        CHECK_EQ(kENCODING_DICT, import_buff->getTypeInfo().get_compression());
        import_buff->addDictEncodedString(*string_payload_ptr);
        const auto ptr = reinterpret_cast<const char*>(import_buff->getStringDictBuffer());
        const auto size = import_buff->getTypeInfo().get_size() * row_count;
        p.fixed_len_data = std::string(ptr, size);
      }
    } else {
      CHECK(import_buff->getTypeInfo().get_type() == kARRAY);
      std::vector<ArrayDatum>* arrays_ptr{nullptr};
      if (IS_STRING(import_buff->getTypeInfo().get_subtype())) {
        CHECK(import_buff->getTypeInfo().get_compression() == kENCODING_DICT);
        import_buff->addDictEncodedStringArray(*import_buff->getStringArrayBuffer());
        arrays_ptr = import_buff->getStringArrayDictBuffer();
      } else {
        arrays_ptr = import_buff->getArrayBuffer();
      }
      for (const auto& arr_datum : *arrays_ptr) {
        TVarLen t_arr_datum;
        if (arr_datum.is_null) {
          t_arr_datum.is_null = true;
        } else {
          CHECK(arr_datum.pointer);
          t_arr_datum.payload = std::string(reinterpret_cast<const char*>(arr_datum.pointer), arr_datum.length);
          t_arr_datum.is_null = false;
        }
        p.var_len_data.push_back(t_arr_datum);
      }
    }
    thrift_insert_data.data.push_back(p);
  }
  if (table_desc->partitions == "REPLICATED") {
    for (size_t leaf_idx = 0; leaf_idx < aggregator_->leafCount(); ++leaf_idx) {
      try {
        aggregator_->insertDataToLeaf(parent_session_info_, leaf_idx, thrift_insert_data);
      } catch (const std::exception& e) {
        LOG(ERROR) << "DistributedLoader Insert Exception: " << e.what();
        return false;
      }
    }
    {
      std::lock_guard<std::mutex> lock(leaves_pending_checkpoint_mutex_);
      for (size_t leaf_idx = 0; leaf_idx < aggregator_->leafCount(); ++leaf_idx) {
        leaves_pending_checkpoint_.insert(leaf_idx);
      }
    }
    return true;
  }
  const auto old_call_count = load_call_count_.fetch_add(1);
  const size_t leaf_idx = old_call_count % aggregator_->leafCount();
  try {
    aggregator_->insertDataToLeaf(parent_session_info_, leaf_idx, thrift_insert_data);
  } catch (const std::exception& e) {
    LOG(ERROR) << "DistributedLoader Insert Exception: " << e.what();
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(leaves_pending_checkpoint_mutex_);
    leaves_pending_checkpoint_.insert(leaf_idx);
  }
  return true;
}
