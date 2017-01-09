#include "ResultSet.h"
#include "GroupByAndAggregate.h"
#include "gen-cpp/serialized_result_set_types.h"
#include "../MapDServer.h"
#include "../Shared/scope.h"

#include <boost/smart_ptr/make_shared.hpp>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

namespace {

#define THRIFT_LAYOUT_CASE(layout)  \
  case GroupByColRangeType::layout: \
    return TResultSetLayout::layout;

TResultSetLayout::type layout_to_thrift(const GroupByColRangeType layout) {
  switch (layout) {
    THRIFT_LAYOUT_CASE(OneColKnownRange)
    THRIFT_LAYOUT_CASE(MultiColPerfectHash)
    THRIFT_LAYOUT_CASE(MultiCol)
    THRIFT_LAYOUT_CASE(Projection)
    THRIFT_LAYOUT_CASE(Scan)
    default:
      CHECK(false);
  }
  abort();
}

#undef THRIFT_LAYOUT_CASE

#define UNTHRIFT_LAYOUT_CASE(layout) \
  case TResultSetLayout::layout:     \
    return GroupByColRangeType::layout;

GroupByColRangeType layout_from_thrift(const TResultSetLayout::type layout) {
  switch (layout) {
    UNTHRIFT_LAYOUT_CASE(OneColKnownRange)
    UNTHRIFT_LAYOUT_CASE(MultiColPerfectHash)
    UNTHRIFT_LAYOUT_CASE(MultiCol)
    UNTHRIFT_LAYOUT_CASE(Projection)
    UNTHRIFT_LAYOUT_CASE(Scan)
    default:
      CHECK(false);
  }
  abort();
}

#undef UNTHRIFT_LAYOUT_CASE

#define THRIFT_AGGKIND_CASE(kind) \
  case k##kind:                   \
    return TAggKind::kind;

TAggKind::type agg_kind_to_thrift(const SQLAgg agg) {
  switch (agg) {
    THRIFT_AGGKIND_CASE(AVG)
    THRIFT_AGGKIND_CASE(COUNT)
    THRIFT_AGGKIND_CASE(MAX)
    THRIFT_AGGKIND_CASE(MIN)
    THRIFT_AGGKIND_CASE(SUM)
    default:
      CHECK(false);
  }
  abort();
}

#undef THRIFT_AGGKIND_CASE

#define UNTHRIFT_AGGKIND_CASE(kind) \
  case TAggKind::kind:              \
    return k##kind;

SQLAgg agg_kind_from_thrift(const TAggKind::type agg) {
  switch (agg) {
    UNTHRIFT_AGGKIND_CASE(AVG)
    UNTHRIFT_AGGKIND_CASE(COUNT)
    UNTHRIFT_AGGKIND_CASE(MAX)
    UNTHRIFT_AGGKIND_CASE(MIN)
    UNTHRIFT_AGGKIND_CASE(SUM)
    default:
      CHECK(false);
  }
  abort();
}

#undef UNTHRIFT_AGGKIND_CASE

TTypeInfo type_info_to_thrift(const SQLTypeInfo& ti) {
  TTypeInfo thrift_ti;
  thrift_ti.type = ti.is_array() ? type_to_thrift(ti.get_elem_type()) : type_to_thrift(ti);
  thrift_ti.encoding = encoding_to_thrift(ti);
  thrift_ti.nullable = !ti.get_notnull();
  thrift_ti.is_array = ti.is_array();
  thrift_ti.precision = ti.get_precision();
  thrift_ti.scale = ti.get_scale();
  thrift_ti.comp_param = ti.get_comp_param();
  return thrift_ti;
}

bool takes_arg(const TargetInfo& target_info) {
  return target_info.is_agg && (target_info.agg_kind != kCOUNT || target_info.is_distinct);
}

TTargetInfo target_info_to_thrift(const TargetInfo& target_info) {
  TTargetInfo thrift_target_info;
  thrift_target_info.is_agg = target_info.is_agg;
  thrift_target_info.kind = agg_kind_to_thrift(target_info.agg_kind);
  thrift_target_info.type = type_info_to_thrift(target_info.sql_type);
  thrift_target_info.arg_type =
      takes_arg(target_info) ? type_info_to_thrift(target_info.agg_arg_type) : thrift_target_info.type;
  thrift_target_info.skip_nulls = target_info.skip_null_val;
  thrift_target_info.is_distinct = target_info.is_distinct;
  return thrift_target_info;
}

TargetInfo target_info_from_thrift(const TTargetInfo& thrift_target_info) {
  TargetInfo target_info;
  target_info.is_agg = thrift_target_info.is_agg;
  target_info.agg_kind = agg_kind_from_thrift(thrift_target_info.kind);
  target_info.sql_type = type_info_from_thrift(thrift_target_info.type);
  target_info.agg_arg_type =
      takes_arg(target_info) ? type_info_from_thrift(thrift_target_info.arg_type) : SQLTypeInfo(kNULLT, false);
  target_info.skip_null_val = thrift_target_info.skip_nulls;
  target_info.is_distinct = thrift_target_info.is_distinct;
  return target_info;
}

std::vector<TTargetInfo> target_infos_to_thrift(const std::vector<TargetInfo>& targets) {
  std::vector<TTargetInfo> thrift_targets;
  for (const auto& target_info : targets) {
    thrift_targets.push_back(target_info_to_thrift(target_info));
  }
  return thrift_targets;
}

std::vector<TargetInfo> target_infos_from_thrift(const std::vector<TTargetInfo>& thrift_targets) {
  std::vector<TargetInfo> targets;
  for (const auto& thrift_target_info : thrift_targets) {
    targets.push_back(target_info_from_thrift(thrift_target_info));
  }
  return targets;
}

TResultSetBufferDescriptor query_mem_desc_to_thrift(const QueryMemoryDescriptor& query_mem_desc) {
  TResultSetBufferDescriptor thrift_query_mem_desc;
  thrift_query_mem_desc.layout = layout_to_thrift(query_mem_desc.hash_type);
  thrift_query_mem_desc.keyless = query_mem_desc.keyless_hash;
  thrift_query_mem_desc.entry_count = query_mem_desc.entry_count;
  thrift_query_mem_desc.min_val = query_mem_desc.min_val;
  thrift_query_mem_desc.max_val = query_mem_desc.max_val;
  thrift_query_mem_desc.bucket = query_mem_desc.bucket;
  for (const auto group_col_width : query_mem_desc.group_col_widths) {
    thrift_query_mem_desc.group_col_widths.push_back(group_col_width);
  }
  for (const auto& agg_col_width : query_mem_desc.agg_col_widths) {
    TColWidths col_widths;
    col_widths.actual = agg_col_width.actual;
    col_widths.compact = agg_col_width.compact;
    thrift_query_mem_desc.agg_col_widths.emplace_back(col_widths);
  }
  for (const auto target_groupby_index : query_mem_desc.target_groupby_indices) {
    thrift_query_mem_desc.target_groupby_indices.push_back(target_groupby_index);
  }
  return thrift_query_mem_desc;
}

QueryMemoryDescriptor query_mem_desc_from_thrift(const TResultSetBufferDescriptor& thrift_query_mem_desc) {
  QueryMemoryDescriptor query_mem_desc{};
  query_mem_desc.hash_type = layout_from_thrift(thrift_query_mem_desc.layout);
  query_mem_desc.keyless_hash = thrift_query_mem_desc.keyless;
  query_mem_desc.interleaved_bins_on_gpu = false;
  query_mem_desc.entry_count = thrift_query_mem_desc.entry_count;
  query_mem_desc.entry_count_small = 0;
  query_mem_desc.min_val = thrift_query_mem_desc.min_val;
  query_mem_desc.max_val = thrift_query_mem_desc.max_val;
  query_mem_desc.bucket = thrift_query_mem_desc.bucket;
  for (const auto group_col_width : thrift_query_mem_desc.group_col_widths) {
    query_mem_desc.group_col_widths.push_back(group_col_width);
  }
  for (const auto& agg_col_width : thrift_query_mem_desc.agg_col_widths) {
    query_mem_desc.agg_col_widths.emplace_back(
        ColWidths{static_cast<int8_t>(agg_col_width.actual), static_cast<int8_t>(agg_col_width.compact)});
  }
  for (const auto target_groupby_index : thrift_query_mem_desc.target_groupby_indices) {
    query_mem_desc.target_groupby_indices.push_back(target_groupby_index);
  }
  return query_mem_desc;
}

}  // namespace

std::string ResultSet::serialize() const {
  if (query_mem_desc_.hash_type == GroupByColRangeType::Projection) {
    return serializeProjection();
  }
  auto buffer = boost::make_shared<apache::thrift::transport::TMemoryBuffer>();
  auto proto = boost::make_shared<apache::thrift::protocol::TBinaryProtocol>(buffer);
  TSerializedRows serialized_rows;
  if (storage_) {
    const auto storage_buffer = reinterpret_cast<const char*>(storage_->getUnderlyingBuffer());
    serialized_rows.buffer = std::string(storage_buffer, storage_->query_mem_desc_.getBufferSizeBytes(device_type_));
  }
  serialized_rows.descriptor = query_mem_desc_to_thrift(query_mem_desc_);
  serialized_rows.targets = target_infos_to_thrift(targets_);
  serialized_rows.write(proto.get());
  return buffer->getBufferAsString();
}

namespace {

void serialize_projected_column(int8_t* col_ptr,
                                std::vector<std::string>& none_encoded_strings,
                                const TargetValue& tv,
                                const SQLTypeInfo& ti) {
  if (ti.is_string() && ti.get_compression() == kENCODING_NONE) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    const auto nullable_str = boost::get<NullableString>(scalar_tv);
    const auto str_p = boost::get<std::string>(nullable_str);
    if (str_p) {
      *reinterpret_cast<int32_t*>(col_ptr) = none_encoded_strings.size();
      none_encoded_strings.push_back(*str_p);
    } else {
      *reinterpret_cast<int32_t*>(col_ptr) = -1;
    }
    return;
  }
  int64_t int_val{0};
  if (ti.is_integer() || ti.is_decimal() || ti.is_boolean() || ti.is_time() || ti.is_string()) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    const auto i64_p = boost::get<int64_t>(scalar_tv);
    CHECK(i64_p);
    int_val = *i64_p;
  }
  double double_val{0};
  float float_val{0};
  if (ti.is_fp()) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    if (ti.get_type() == kDOUBLE) {
      const auto double_p = boost::get<double>(scalar_tv);
      CHECK(double_p);
      double_val = *double_p;
    } else {
      CHECK_EQ(kFLOAT, ti.get_type());
      const auto float_p = boost::get<float>(scalar_tv);
      CHECK(float_p);
      float_val = *float_p;
    }
  }
  switch (ti.get_type()) {
    case kBOOLEAN: {
      *reinterpret_cast<int8_t*>(col_ptr) = int_val;
      break;
    }
    case kSMALLINT: {
      *reinterpret_cast<int16_t*>(col_ptr) = int_val;
      break;
    }
    case kINT:
    case kCHAR:
    case kVARCHAR:
    case kTEXT: {
      *reinterpret_cast<int32_t*>(col_ptr) = int_val;
      break;
    }
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kDATE:
    case kTIMESTAMP:
    case kTIME: {
      *reinterpret_cast<int64_t*>(col_ptr) = int_val;
      break;
    }
    case kFLOAT: {
      *reinterpret_cast<float*>(col_ptr) = float_val;
      break;
    }
    case kDOUBLE: {
      *reinterpret_cast<double*>(col_ptr) = double_val;
      break;
    }
    default:
      CHECK(false);
  }
}

}  // namespace

// The projection layout could contain lazy values, real strings and arrays.
// Leverage the high-level row iteration to retrieve the values and put them
// in the address-space independent buffer which then goes over the network.
std::string ResultSet::serializeProjection() const {
  moveToBegin();
  ScopeGuard restore_cursor = [this] { moveToBegin(); };
  size_t one_row_size{8};  // Store the index of the row for now, although it's
                           // redundant since we only serialize non-empty entries.
  auto proj_query_mem_desc = query_mem_desc_;
  proj_query_mem_desc.agg_col_widths.clear();
  for (size_t i = 0; i < colCount(); ++i) {
    const auto ti = getColType(i);
    const int8_t logical_size = (ti.is_string() && ti.get_compression() == kENCODING_NONE) ? 4 : ti.get_logical_size();
    proj_query_mem_desc.agg_col_widths.emplace_back(ColWidths{logical_size, logical_size});
    one_row_size += logical_size;
  }
  std::unique_ptr<int8_t[]> serialized_storage(new int8_t[one_row_size * entryCount()]);
  auto row_ptr = serialized_storage.get();
  size_t row_count{0};
  std::vector<std::string> none_encoded_strings;
  while (true) {
    const auto crt_row = getNextRow(false, false);
    if (crt_row.empty()) {
      break;
    }
    CHECK_EQ(colCount(), crt_row.size());
    *reinterpret_cast<int64_t*>(row_ptr) = row_count;
    auto col_ptr = row_ptr + 8;
    for (size_t i = 0; i < colCount(); ++i) {
      const auto ti = getColType(i);
      serialize_projected_column(col_ptr, none_encoded_strings, crt_row[i], ti);
      col_ptr += ti.get_logical_size();
    }
    ++row_count;
    row_ptr += one_row_size;
  }
  proj_query_mem_desc.entry_count = row_count;
  TSerializedRows serialized_rows;
  serialized_rows.buffer =
      std::string(reinterpret_cast<const char*>(serialized_storage.get()), one_row_size * row_count);
  serialized_rows.descriptor = query_mem_desc_to_thrift(proj_query_mem_desc);
  serialized_rows.targets = target_infos_to_thrift(targets_);
  serialized_rows.none_encoded_strings.swap(none_encoded_strings);
  auto buffer = boost::make_shared<apache::thrift::transport::TMemoryBuffer>();
  auto proto = boost::make_shared<apache::thrift::protocol::TBinaryProtocol>(buffer);
  serialized_rows.write(proto.get());
  return buffer->getBufferAsString();
}

std::unique_ptr<ResultSet> ResultSet::unserialize(const std::string& str) {
  auto buffer_bytes = reinterpret_cast<uint8_t*>(const_cast<char*>(str.data()));
  auto buffer = boost::make_shared<apache::thrift::transport::TMemoryBuffer>(buffer_bytes, str.size());
  auto proto = boost::make_shared<apache::thrift::protocol::TBinaryProtocol>(buffer);
  TSerializedRows serialized_rows;
  serialized_rows.read(proto.get());
  const auto target_infos = target_infos_from_thrift(serialized_rows.targets);
  const auto query_mem_desc = query_mem_desc_from_thrift(serialized_rows.descriptor);
  auto result_set =
      boost::make_unique<ResultSet>(target_infos, ExecutorDeviceType::CPU, query_mem_desc, nullptr, nullptr);
  if (query_mem_desc.entry_count) {
    auto storage = result_set->allocateStorage();
    auto storage_buff = storage->getUnderlyingBuffer();
    memcpy(storage_buff, serialized_rows.buffer.data(), serialized_rows.buffer.size());
  }
  result_set->none_encoded_strings_valid_ = true;
  result_set->none_encoded_strings_.emplace_back(std::move(serialized_rows.none_encoded_strings));
  return std::move(result_set);
}
