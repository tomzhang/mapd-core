include "mapd.thrift"

enum TResultSetLayout {
  OneColKnownRange,
  MultiColPerfectHash,
  MultiCol,
  Projection,
  Scan
}

struct TColWidths {
  1: i16 actual,
  2: i16 compact
}

struct TResultSetBufferDescriptor {
  1: TResultSetLayout layout,
  2: bool keyless,
  3: i32 entry_count,
  4: i32 idx_target_as_key,
  5: i64 init_val,
  6: i64 min_val,
  7: i64 max_val,
  8: i64 bucket,
  9: list<i16> group_col_widths,
  10: list<TColWidths> agg_col_widths,
  11: list<i32> target_groupby_indices
}

enum TAggKind {
  AVG, MIN, MAX, SUM, COUNT
}

struct TTargetInfo {
  1: bool is_agg,
  2: TAggKind kind,
  3: mapd.TTypeInfo type,
  4: mapd.TTypeInfo arg_type,
  5: bool skip_nulls,
  6: bool is_distinct
}

struct TSerializedRows {
  1: binary buffer,
  2: TResultSetBufferDescriptor descriptor,
  3: list<TTargetInfo> targets,
  4: list<string> none_encoded_strings
}
