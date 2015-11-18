#include "RuntimeFunctions.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <tuple>

// decoder implementations

#include "DecodersImpl.h"

// arithmetic operator implementations

#define DEF_ARITH_NULLABLE(type, null_type, opname, opsym)                                        \
  extern "C" __attribute__((always_inline))                                                       \
      type opname##_##type##_nullable(const type lhs, const type rhs, const null_type null_val) { \
    if (lhs != null_val && rhs != null_val) {                                                     \
      return lhs opsym rhs;                                                                       \
    }                                                                                             \
    return null_val;                                                                              \
  }

#define DEF_ARITH_NULLABLE_LHS(type, null_type, opname, opsym)                                        \
  extern "C" __attribute__((always_inline))                                                           \
      type opname##_##type##_nullable_lhs(const type lhs, const type rhs, const null_type null_val) { \
    if (lhs != null_val) {                                                                            \
      return lhs opsym rhs;                                                                           \
    }                                                                                                 \
    return null_val;                                                                                  \
  }

#define DEF_ARITH_NULLABLE_RHS(type, null_type, opname, opsym)                                        \
  extern "C" __attribute__((always_inline))                                                           \
      type opname##_##type##_nullable_rhs(const type lhs, const type rhs, const null_type null_val) { \
    if (rhs != null_val) {                                                                            \
      return lhs opsym rhs;                                                                           \
    }                                                                                                 \
    return null_val;                                                                                  \
  }

#define DEF_CMP_NULLABLE(type, null_type, opname, opsym)                                      \
  extern "C" __attribute__((always_inline)) int8_t opname##_##type##_nullable(                \
      const type lhs, const type rhs, const null_type null_val, const int8_t null_bool_val) { \
    if (lhs != null_val && rhs != null_val) {                                                 \
      return lhs opsym rhs;                                                                   \
    }                                                                                         \
    return null_bool_val;                                                                     \
  }

#define DEF_CMP_NULLABLE_LHS(type, null_type, opname, opsym)                                  \
  extern "C" __attribute__((always_inline)) int8_t opname##_##type##_nullable_lhs(            \
      const type lhs, const type rhs, const null_type null_val, const int8_t null_bool_val) { \
    if (lhs != null_val) {                                                                    \
      return lhs opsym rhs;                                                                   \
    }                                                                                         \
    return null_bool_val;                                                                     \
  }

#define DEF_CMP_NULLABLE_RHS(type, null_type, opname, opsym)                                  \
  extern "C" __attribute__((always_inline)) int8_t opname##_##type##_nullable_rhs(            \
      const type lhs, const type rhs, const null_type null_val, const int8_t null_bool_val) { \
    if (rhs != null_val) {                                                                    \
      return lhs opsym rhs;                                                                   \
    }                                                                                         \
    return null_bool_val;                                                                     \
  }

#define DEF_BINARY_NULLABLE_ALL_OPS(type, null_type) \
  DEF_ARITH_NULLABLE(type, null_type, add, +)        \
  DEF_ARITH_NULLABLE(type, null_type, sub, -)        \
  DEF_ARITH_NULLABLE(type, null_type, mul, *)        \
  DEF_ARITH_NULLABLE(type, null_type, div, / )       \
  DEF_ARITH_NULLABLE_LHS(type, null_type, add, +)    \
  DEF_ARITH_NULLABLE_LHS(type, null_type, sub, -)    \
  DEF_ARITH_NULLABLE_LHS(type, null_type, mul, *)    \
  DEF_ARITH_NULLABLE_LHS(type, null_type, div, / )   \
  DEF_ARITH_NULLABLE_RHS(type, null_type, add, +)    \
  DEF_ARITH_NULLABLE_RHS(type, null_type, sub, -)    \
  DEF_ARITH_NULLABLE_RHS(type, null_type, mul, *)    \
  DEF_ARITH_NULLABLE_RHS(type, null_type, div, / )   \
  DEF_CMP_NULLABLE(type, null_type, eq, == )         \
  DEF_CMP_NULLABLE(type, null_type, ne, != )         \
  DEF_CMP_NULLABLE(type, null_type, lt, < )          \
  DEF_CMP_NULLABLE(type, null_type, gt, > )          \
  DEF_CMP_NULLABLE(type, null_type, le, <= )         \
  DEF_CMP_NULLABLE(type, null_type, ge, >= )         \
  DEF_CMP_NULLABLE_LHS(type, null_type, eq, == )     \
  DEF_CMP_NULLABLE_LHS(type, null_type, ne, != )     \
  DEF_CMP_NULLABLE_LHS(type, null_type, lt, < )      \
  DEF_CMP_NULLABLE_LHS(type, null_type, gt, > )      \
  DEF_CMP_NULLABLE_LHS(type, null_type, le, <= )     \
  DEF_CMP_NULLABLE_LHS(type, null_type, ge, >= )     \
  DEF_CMP_NULLABLE_RHS(type, null_type, eq, == )     \
  DEF_CMP_NULLABLE_RHS(type, null_type, ne, != )     \
  DEF_CMP_NULLABLE_RHS(type, null_type, lt, < )      \
  DEF_CMP_NULLABLE_RHS(type, null_type, gt, > )      \
  DEF_CMP_NULLABLE_RHS(type, null_type, le, <= )     \
  DEF_CMP_NULLABLE_RHS(type, null_type, ge, >= )

DEF_BINARY_NULLABLE_ALL_OPS(int16_t, int64_t)
DEF_BINARY_NULLABLE_ALL_OPS(int32_t, int64_t)
DEF_BINARY_NULLABLE_ALL_OPS(int64_t, int64_t)
DEF_BINARY_NULLABLE_ALL_OPS(float, float)
DEF_BINARY_NULLABLE_ALL_OPS(double, double)
DEF_CMP_NULLABLE(int8_t, int64_t, eq, == )
DEF_CMP_NULLABLE(int8_t, int64_t, ne, != )
DEF_CMP_NULLABLE_LHS(int8_t, int64_t, eq, == )
DEF_CMP_NULLABLE_LHS(int8_t, int64_t, ne, != )
DEF_CMP_NULLABLE_RHS(int8_t, int64_t, eq, == )
DEF_CMP_NULLABLE_RHS(int8_t, int64_t, ne, != )
DEF_ARITH_NULLABLE(int8_t, int64_t, mod, % )
DEF_ARITH_NULLABLE(int16_t, int64_t, mod, % )
DEF_ARITH_NULLABLE(int32_t, int64_t, mod, % )
DEF_ARITH_NULLABLE(int64_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_LHS(int8_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_LHS(int16_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_LHS(int32_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_LHS(int64_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_RHS(int8_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_RHS(int16_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_RHS(int32_t, int64_t, mod, % )
DEF_ARITH_NULLABLE_RHS(int64_t, int64_t, mod, % )

#undef DEF_BINARY_NULLABLE_ALL_OPS
#undef DEF_CMP_NULLABLE_RHS
#undef DEF_CMP_NULLABLE_LHS
#undef DEF_CMP_NULLABLE
#undef DEF_ARITH_NULLABLE_RHS
#undef DEF_ARITH_NULLABLE_LHS
#undef DEF_ARITH_NULLABLE

#define DEF_MUL_DECIMAL(type)                                                                                \
  extern "C" __attribute__((always_inline))                                                                  \
      type mul_##type##_decimal(const type lhs, const type rhs, const uint64_t scale, const type null_val) { \
    if (lhs != null_val && rhs != null_val) {                                                                \
      return (static_cast<double>(lhs) * rhs) / scale;                                                       \
    }                                                                                                        \
    return null_val;                                                                                         \
  }

DEF_MUL_DECIMAL(int16_t)
DEF_MUL_DECIMAL(int32_t)
DEF_MUL_DECIMAL(int64_t)

#undef DEF_MUL_DECIMAL

extern "C" __attribute__((always_inline)) int64_t scale_decimal(const int64_t operand,
                                                                const uint64_t scale,
                                                                const int64_t operand_null_val,
                                                                const int64_t result_null_val) {
  return operand != operand_null_val ? operand * scale : result_null_val;
}

#define DEF_UMINUS_NULLABLE(type, null_type)                                        \
  extern "C" __attribute__((always_inline))                                         \
      type uminus_##type##_nullable(const type operand, const null_type null_val) { \
    return operand == null_val ? null_val : -operand;                               \
  }

DEF_UMINUS_NULLABLE(int16_t, int64_t)
DEF_UMINUS_NULLABLE(int32_t, int64_t)
DEF_UMINUS_NULLABLE(int64_t, int64_t)
DEF_UMINUS_NULLABLE(float, float)
DEF_UMINUS_NULLABLE(double, double)

#undef DEF_UMINUS_NULLABLE

#define DEF_CAST_NULLABLE(from_type, to_type)                                                   \
  extern "C" __attribute__((always_inline)) to_type cast_##from_type##_to_##to_type##_nullable( \
      const from_type operand, const from_type from_null_val, const to_type to_null_val) {      \
    return operand == from_null_val ? to_null_val : operand;                                    \
  }

#define DEF_CAST_NULLABLE_BIDIR(type1, type2) \
  DEF_CAST_NULLABLE(type1, type2)             \
  DEF_CAST_NULLABLE(type2, type1)

DEF_CAST_NULLABLE_BIDIR(int8_t, int16_t)
DEF_CAST_NULLABLE_BIDIR(int8_t, int32_t)
DEF_CAST_NULLABLE_BIDIR(int8_t, int64_t)
DEF_CAST_NULLABLE_BIDIR(int16_t, int32_t)
DEF_CAST_NULLABLE_BIDIR(int16_t, int64_t)
DEF_CAST_NULLABLE_BIDIR(int32_t, int64_t)
DEF_CAST_NULLABLE_BIDIR(float, double)
DEF_CAST_NULLABLE_BIDIR(float, int8_t)
DEF_CAST_NULLABLE_BIDIR(float, int16_t)
DEF_CAST_NULLABLE_BIDIR(float, int32_t)
DEF_CAST_NULLABLE_BIDIR(float, int64_t)
DEF_CAST_NULLABLE_BIDIR(double, int8_t)
DEF_CAST_NULLABLE_BIDIR(double, int16_t)
DEF_CAST_NULLABLE_BIDIR(double, int32_t)
DEF_CAST_NULLABLE_BIDIR(double, int64_t)

#undef DEF_CAST_NULLABLE_BIDIR
#undef DEF_CAST_NULLABLE

extern "C" __attribute__((always_inline)) int8_t logical_not(const int8_t operand, const int8_t null_val) {
  return operand == null_val ? operand : (operand ? 0 : 1);
}

extern "C" __attribute__((always_inline)) int8_t
    logical_and(const int8_t lhs, const int8_t rhs, const int8_t null_val) {
  if (lhs == null_val || rhs == null_val) {
    return null_val;
  }
  return (lhs && rhs) ? 1 : 0;
}

extern "C" __attribute__((always_inline)) int8_t logical_or(const int8_t lhs, const int8_t rhs, const int8_t null_val) {
  if (lhs == null_val || rhs == null_val) {
    return null_val;
  }
  return (lhs || rhs) ? 1 : 0;
}

#define DEF_TRANSLATE_NULL_KEY(key_type)                                            \
  extern "C" __attribute__((always_inline)) int64_t translate_null_key_##key_type(  \
      const key_type key, const key_type null_val, const key_type translated_val) { \
    if (key == null_val) {                                                          \
      return translated_val;                                                        \
    }                                                                               \
    return key;                                                                     \
  }

DEF_TRANSLATE_NULL_KEY(int8_t)
DEF_TRANSLATE_NULL_KEY(int16_t)
DEF_TRANSLATE_NULL_KEY(int32_t)
DEF_TRANSLATE_NULL_KEY(int64_t)

#undef DEF_TRANSLATE_NULL_KEY

// aggregator implementations

extern "C" __attribute__((always_inline)) void agg_count(int64_t* agg, const int64_t val) {
  ++*agg;
}

extern "C" __attribute__((always_inline)) void agg_count_distinct(int64_t* agg, const int64_t val) {
  reinterpret_cast<std::set<int64_t>*>(*agg)->insert(val);
}

extern "C" __attribute__((always_inline)) void agg_count_distinct_bitmap(int64_t* agg,
                                                                         const int64_t val,
                                                                         const int64_t min_val) {
  const uint64_t bitmap_idx = val - min_val;
  reinterpret_cast<int8_t*>(*agg)[bitmap_idx >> 3] |= (1 << (bitmap_idx & 7));
}

extern "C" __attribute__((always_inline)) void agg_sum(int64_t* agg, const int64_t val) {
  *agg += val;
}

extern "C" __attribute__((always_inline)) void agg_max(int64_t* agg, const int64_t val) {
  *agg = std::max(*agg, val);
}

extern "C" __attribute__((always_inline)) void agg_min(int64_t* agg, const int64_t val) {
  *agg = std::min(*agg, val);
}

extern "C" __attribute__((always_inline)) void agg_id(int64_t* agg, const int64_t val) {
  *agg = val;
}

extern "C" __attribute__((always_inline)) void agg_count_distinct_skip_val(int64_t* agg,
                                                                           const int64_t val,
                                                                           const int64_t skip_val) {
  if (val != skip_val) {
    agg_count_distinct(agg, val);
  }
}

extern "C" __attribute__((always_inline)) void agg_count_distinct_bitmap_skip_val(int64_t* agg,
                                                                                  const int64_t val,
                                                                                  const int64_t min_val,
                                                                                  const int64_t skip_val) {
  if (val != skip_val) {
    agg_count_distinct_bitmap(agg, val, min_val);
  }
}

#define DEF_SKIP_AGG_ADD(base_agg_func)                                    \
  extern "C" __attribute__((always_inline)) void base_agg_func##_skip_val( \
      int64_t* agg, const int64_t val, const int64_t skip_val) {           \
    if (val != skip_val) {                                                 \
      base_agg_func(agg, val);                                             \
    }                                                                      \
  }

#define DEF_SKIP_AGG(base_agg_func)                                        \
  extern "C" __attribute__((always_inline)) void base_agg_func##_skip_val( \
      int64_t* agg, const int64_t val, const int64_t skip_val) {           \
    if (val != skip_val) {                                                 \
      const int64_t old_agg = *agg;                                        \
      if (old_agg != skip_val) {                                           \
        base_agg_func(agg, val);                                           \
      } else {                                                             \
        *agg = val;                                                        \
      }                                                                    \
    }                                                                      \
  }

DEF_SKIP_AGG_ADD(agg_count)
DEF_SKIP_AGG_ADD(agg_sum)
DEF_SKIP_AGG(agg_max)
DEF_SKIP_AGG(agg_min)

#undef DEF_SKIP_AGG_ADD
#undef DEF_SKIP_AGG

// TODO(alex): fix signature

extern "C" __attribute__((always_inline)) void agg_count_double(int64_t* agg, const double val) {
  ++*agg;
}

extern "C" __attribute__((always_inline)) void agg_sum_double(int64_t* agg, const double val) {
  const auto r = *reinterpret_cast<const double*>(agg) + val;
  *agg = *reinterpret_cast<const int64_t*>(&r);
}

extern "C" __attribute__((always_inline)) void agg_max_double(int64_t* agg, const double val) {
  const auto r = std::max(*reinterpret_cast<const double*>(agg), val);
  *agg = *(reinterpret_cast<const int64_t*>(&r));
}

extern "C" __attribute__((always_inline)) void agg_min_double(int64_t* agg, const double val) {
  const auto r = std::min(*reinterpret_cast<const double*>(agg), val);
  *agg = *(reinterpret_cast<const int64_t*>(&r));
}

extern "C" __attribute__((always_inline)) void agg_id_double(int64_t* agg, const double val) {
  *agg = *(reinterpret_cast<const int64_t*>(&val));
}

#define DEF_SKIP_AGG_ADD(base_agg_func)                                    \
  extern "C" __attribute__((always_inline)) void base_agg_func##_skip_val( \
      int64_t* agg, const double val, const double skip_val) {             \
    if (val != skip_val) {                                                 \
      base_agg_func(agg, val);                                             \
    }                                                                      \
  }

#define DEF_SKIP_AGG(base_agg_func)                                        \
  extern "C" __attribute__((always_inline)) void base_agg_func##_skip_val( \
      int64_t* agg, const double val, const double skip_val) {             \
    if (val != skip_val) {                                                 \
      const int64_t old_agg = *agg;                                        \
      if (old_agg != *reinterpret_cast<const int64_t*>(&skip_val)) {       \
        base_agg_func(agg, val);                                           \
      } else {                                                             \
        *agg = *reinterpret_cast<const int64_t*>(&val);                    \
      }                                                                    \
    }                                                                      \
  }

DEF_SKIP_AGG_ADD(agg_count_double)
DEF_SKIP_AGG_ADD(agg_sum_double)
DEF_SKIP_AGG(agg_max_double)
DEF_SKIP_AGG(agg_min_double)

#undef DEF_SKIP_AGG_ADD
#undef DEF_SKIP_AGG

// Shared memory aggregators. Should never be called,
// real implementations are in cuda_mapd_rt.cu.

#define DEF_SHARED_AGG_STUBS(base_agg_func)                                                                            \
  extern "C" __attribute__((noinline)) void base_agg_func##_shared(int64_t* agg, const int64_t val) { abort(); }       \
                                                                                                                       \
  extern "C" __attribute__((noinline)) void base_agg_func##_skip_val_shared(                                           \
      int64_t* agg, const int64_t val, const int64_t skip_val) {                                                       \
    abort();                                                                                                           \
  }                                                                                                                    \
                                                                                                                       \
  extern "C" __attribute__((noinline)) void base_agg_func##_double_shared(int64_t* agg, const double val) { abort(); } \
                                                                                                                       \
  extern "C" __attribute__((noinline)) void base_agg_func##_double_skip_val_shared(                                    \
      int64_t* agg, const double val, const double skip_val) {                                                         \
    abort();                                                                                                           \
  }

DEF_SHARED_AGG_STUBS(agg_count)
DEF_SHARED_AGG_STUBS(agg_sum)
DEF_SHARED_AGG_STUBS(agg_max)
DEF_SHARED_AGG_STUBS(agg_min)
DEF_SHARED_AGG_STUBS(agg_id)

// x64 stride functions

extern "C" __attribute__((noinline)) int32_t pos_start_impl(const int32_t* row_index_resume) {
  return row_index_resume ? row_index_resume[0] : 0;
}

extern "C" __attribute__((noinline)) int32_t group_buff_idx_impl() {
  return pos_start_impl(nullptr);
}

extern "C" __attribute__((noinline)) int32_t pos_step_impl() {
  return 1;
}

extern "C" __attribute__((noinline)) int8_t thread_warp_idx(const int8_t warp_sz) {
  abort();
}

extern "C" __attribute__((always_inline)) int32_t merge_error_code(const int32_t err_code, int32_t* merged_err_code) {
  if (err_code) {
    *merged_err_code = err_code;
  }
  return err_code;
}

// group by helpers

extern "C" __attribute__((noinline)) const int64_t
    * init_shared_mem_nop(const int64_t* groups_buffer, const int32_t groups_buffer_size) {
  return groups_buffer;
}

extern "C" __attribute__((noinline)) void write_back_nop(int64_t* dest, int64_t* src, const int32_t sz) {
  // the body is not really needed, just make sure the call is not optimized away
  assert(dest);
}

extern "C" __attribute__((noinline)) const int64_t
    * init_shared_mem(const int64_t* groups_buffer, const int32_t groups_buffer_size) {
  return init_shared_mem_nop(groups_buffer, groups_buffer_size);
}

extern "C" __attribute__((noinline)) void write_back(int64_t* dest, int64_t* src, const int32_t sz) {
  write_back_nop(dest, src, sz);
}

extern "C" __attribute__((noinline)) void init_group_by_buffer_gpu(int64_t* groups_buffer,
                                                                   const int64_t* init_vals,
                                                                   const uint32_t groups_buffer_entry_count,
                                                                   const uint32_t key_qw_count,
                                                                   const uint32_t agg_col_count,
                                                                   const bool keyless,
                                                                   const int8_t warp_size) {
  // the body is not really needed, just make sure the call is not optimized away
  assert(groups_buffer);
}

extern "C" __attribute__((noinline)) void init_columnar_group_by_buffer_gpu(int64_t* groups_buffer,
                                                                            const int64_t* init_vals,
                                                                            const uint32_t groups_buffer_entry_count,
                                                                            const uint32_t key_qw_count,
                                                                            const uint32_t agg_col_count,
                                                                            const bool keyless,
                                                                            const bool blocks_share_memory,
                                                                            const int32_t frag_idx) {
  // the body is not really needed, just make sure the call is not optimized away
  assert(groups_buffer);
}

extern "C" __attribute__((noinline)) void init_group_by_buffer_impl(int64_t* groups_buffer,
                                                                    const int64_t* init_vals,
                                                                    const uint32_t groups_buffer_entry_count,
                                                                    const uint32_t key_qw_count,
                                                                    const uint32_t agg_col_count,
                                                                    const bool keyless,
                                                                    const int8_t warp_size) {
  // the body is not really needed, just make sure the call is not optimized away
  assert(groups_buffer);
}

extern "C" __attribute__((always_inline)) int64_t* get_matching_group_value(int64_t* groups_buffer,
                                                                            const uint32_t h,
                                                                            const int64_t* key,
                                                                            const uint32_t key_qw_count,
                                                                            const uint32_t agg_col_count,
                                                                            const int64_t* init_vals) {
  auto off = h * (key_qw_count + agg_col_count);
  if (groups_buffer[off] == EMPTY_KEY) {
    memcpy(groups_buffer + off, key, key_qw_count * sizeof(*key));
    return groups_buffer + off + key_qw_count;
  }
  if (memcmp(groups_buffer + off, key, key_qw_count * sizeof(*key)) == 0) {
    return groups_buffer + off + key_qw_count;
  }
  return nullptr;
}

extern "C" __attribute__((always_inline)) int64_t* get_matching_group_value_perfect_hash(int64_t* groups_buffer,
                                                                                         const uint32_t h,
                                                                                         const int64_t* key,
                                                                                         const uint32_t key_qw_count,
                                                                                         const uint32_t agg_col_count) {
  uint32_t off = h * (key_qw_count + agg_col_count);
  if (groups_buffer[off] == EMPTY_KEY) {
    for (uint32_t i = 0; i < key_qw_count; ++i) {
      groups_buffer[off + i] = key[i];
    }
  }
  return groups_buffer + off + key_qw_count;
}

#include "GroupByRuntime.cpp"

extern "C" __attribute__((always_inline)) int64_t* get_group_value_fast_keyless(int64_t* groups_buffer,
                                                                                const int64_t key,
                                                                                const int64_t min_key,
                                                                                const uint32_t agg_col_count) {
  return groups_buffer + agg_col_count * (key - min_key);
}

extern "C" __attribute__((always_inline)) int64_t
    * get_group_value_fast_keyless_semiprivate(int64_t* groups_buffer,
                                               const int64_t key,
                                               const int64_t min_key,
                                               const uint32_t agg_col_count,
                                               const uint8_t thread_warp_idx,
                                               const uint8_t warp_size) {
  return groups_buffer + agg_col_count * (warp_size * (key - min_key) + thread_warp_idx);
}

extern "C" __attribute__((always_inline)) int8_t* extract_str_ptr(const uint64_t str_and_len) {
  return reinterpret_cast<int8_t*>(str_and_len & 0xffffffffffff);
}

extern "C" __attribute__((always_inline)) int32_t extract_str_len(const uint64_t str_and_len) {
  return static_cast<int64_t>(str_and_len) >> 48;
}

extern "C" __attribute__((noinline)) int8_t* extract_str_ptr_noinline(const uint64_t str_and_len) {
  return extract_str_ptr(str_and_len);
}

extern "C" __attribute__((noinline)) int32_t extract_str_len_noinline(const uint64_t str_and_len) {
  return extract_str_len(str_and_len);
}

extern "C" __attribute__((always_inline)) uint64_t string_pack(const int8_t* ptr, const int32_t len) {
  return (reinterpret_cast<const uint64_t>(ptr) & 0xffffffffffff) | (static_cast<const uint64_t>(len) << 48);
}

#ifdef __clang__
#include "../Utils/StringLike.cpp"
#endif

extern "C" ALWAYS_INLINE DEVICE int32_t char_length(const char* str, const int32_t str_len) {
  return str_len;
}

extern "C" ALWAYS_INLINE DEVICE int32_t
    char_length_nullable(const char* str, const int32_t str_len, const int32_t int_null) {
  if (!str) {
    return int_null;
  }
  return str_len;
}

extern "C" __attribute__((noinline)) void query_stub_hoisted_literals(const int8_t** col_buffers,
                                                                      const int8_t* literals,
                                                                      const int64_t* num_rows,
                                                                      const uint64_t* frag_row_offsets,
                                                                      const int64_t* max_matched,
                                                                      const int64_t* init_agg_value,
                                                                      int64_t** out,
                                                                      int64_t** out2,
                                                                      uint32_t frag_idx,
                                                                      int32_t* resume_row_index) {
  assert(col_buffers || literals || num_rows || frag_row_offsets || max_matched || init_agg_value || out || out2 ||
         frag_idx || resume_row_index);
}

extern "C" void multifrag_query_hoisted_literals(const int8_t*** col_buffers,
                                                 const uint32_t* num_fragments,
                                                 const int8_t* literals,
                                                 const int64_t* num_rows,
                                                 const uint64_t* frag_row_offsets,
                                                 const int64_t* max_matched,
                                                 const int64_t* init_agg_value,
                                                 int64_t** out,
                                                 int64_t** out2,
                                                 int32_t* resume_row_index,
                                                 const uint32_t* num_tables_ptr) {
  for (uint32_t i = 0; i < *num_fragments; ++i) {
    query_stub_hoisted_literals(col_buffers ? col_buffers[i] : nullptr,
                                literals,
                                &num_rows[i * (*num_tables_ptr)],
                                &frag_row_offsets[i],
                                max_matched,
                                init_agg_value,
                                out,
                                out2,
                                i,
                                resume_row_index);
  }
}

extern "C" __attribute__((noinline)) void query_stub(const int8_t** col_buffers,
                                                     const int64_t* num_rows,
                                                     const uint64_t* frag_row_offsets,
                                                     const int64_t* max_matched,
                                                     const int64_t* init_agg_value,
                                                     int64_t** out,
                                                     int64_t** out2,
                                                     uint32_t frag_idx,
                                                     int32_t* resume_row_index) {
  assert(col_buffers || num_rows || frag_row_offsets || max_matched || init_agg_value || out || out2 || frag_idx ||
         resume_row_index);
}

extern "C" void multifrag_query(const int8_t*** col_buffers,
                                const uint32_t* num_fragments,
                                const int64_t* num_rows,
                                const uint64_t* frag_row_offsets,
                                const int64_t* max_matched,
                                const int64_t* init_agg_value,
                                int64_t** out,
                                int64_t** out2,
                                int32_t* resume_row_index,
                                const uint32_t* num_tables_ptr) {
  for (uint32_t i = 0; i < *num_fragments; ++i) {
    query_stub(col_buffers ? col_buffers[i] : nullptr,
               &num_rows[i * (*num_tables_ptr)],
               &frag_row_offsets[i],
               max_matched,
               init_agg_value,
               out,
               out2,
               i,
               resume_row_index);
  }
}

extern "C" __attribute__((always_inline)) int64_t
    hash_join_idx(int64_t hash_buff, const int64_t key, const int64_t min_key, const int64_t max_key) {
  if (key >= min_key && key <= max_key) {
    return *get_hash_slot(reinterpret_cast<int64_t*>(hash_buff), key, min_key, 1);
  }
  return -1;
}
