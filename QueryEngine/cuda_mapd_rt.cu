#include <stdint.h>
#include <limits>

extern "C"
__device__ int32_t pos_start_impl(const int32_t* row_index_resume) {
  return blockIdx.x * blockDim.x + threadIdx.x +
    (row_index_resume ? row_index_resume[blockIdx.x] : 0);
}

extern "C"
__device__ int32_t pos_step_impl() {
  return blockDim.x * gridDim.x;
}

extern "C"
__device__ int8_t thread_warp_idx(const int8_t warp_sz) {
  return threadIdx.x % warp_sz;
}

extern "C"
__device__ const int64_t* init_shared_mem_nop(const int64_t* groups_buffer,
                                              const int32_t groups_buffer_size) {
  return groups_buffer;
}

extern "C"
__device__ void write_back_nop(int64_t* dest, int64_t* src, const int32_t sz) {
}

extern "C"
__device__ const int64_t* init_shared_mem(const int64_t* groups_buffer,
                                          const int32_t groups_buffer_size) {
  extern __shared__ int64_t fast_bins[];
  if (threadIdx.x == 0) {
    memcpy(fast_bins, groups_buffer, groups_buffer_size);
  }
  __syncthreads();
  return fast_bins;
}

extern "C"
__device__ void write_back(int64_t* dest, int64_t* src, const int32_t sz) {
  __syncthreads();
  if (threadIdx.x == 0) {
    memcpy(dest, src, sz);
  }
}

#define EMPTY_KEY 9223372036854775807L

extern "C"
__device__ void init_group_by_buffer_impl(int64_t* groups_buffer,
                                          const int64_t* init_vals,
                                          const uint32_t groups_buffer_entry_count,
                                          const uint32_t key_qw_count,
                                          const uint32_t agg_col_count) {
  const int32_t start = threadIdx.x;
  const int32_t step = blockDim.x;
  uint32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + agg_col_count);
  for (uint32_t i = start; i < groups_buffer_entry_qw_count; i += step) {
    groups_buffer[i] = EMPTY_KEY;
  }
  __syncthreads();
}

extern "C"
__device__ int64_t* get_matching_group_value(int64_t* groups_buffer,
                                             const uint32_t h,
                                             const int64_t* key,
                                             const uint32_t key_qw_count,
                                             const uint32_t agg_col_count,
                                             const int64_t* init_vals) {
  uint32_t off = h * (key_qw_count + agg_col_count);
  {
    const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(groups_buffer + off),
      EMPTY_KEY, *key);
    if (EMPTY_KEY == old) {
      memcpy(groups_buffer + off, key, key_qw_count * sizeof(*key));
      memcpy(groups_buffer + off + key_qw_count, init_vals, agg_col_count * sizeof(*init_vals));
    }
  }
  __syncthreads();
  bool match = true;
  for (uint32_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off + i] != key[i]) {
      match = false;
      break;
    }
  }
  return match ? groups_buffer + off + key_qw_count : NULL;
}

#include "GroupByRuntime.cpp"

__device__ int64_t atomicMax64(int64_t* address, int64_t val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        max((long long) val, (long long) assumed));
    } while (assumed != old);

    return old;
}

__device__ int64_t atomicMin64(int64_t* address, int64_t val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        min((long long) val, (long long) assumed));
    } while (assumed != old);

    return old;
}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(max(val,
                               __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(min(val,
                               __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

extern "C"
__device__ void agg_count_shared(int64_t* agg, const int64_t val) {
  atomicAdd(reinterpret_cast<int32_t*>(agg), 1L);
}

extern "C"
__device__ void agg_count_double_shared(int64_t* agg, const double val) {
  agg_count_shared(agg, val);
}

extern "C"
__device__ void agg_sum_shared(int64_t* agg, const int64_t val) {
  atomicAdd(reinterpret_cast<unsigned long long*>(agg), val);
}

extern "C"
__device__ void agg_sum_double_shared(int64_t* agg, const double val) {
  atomicAdd(reinterpret_cast<double*>(agg), val);
}

extern "C"
__device__ void agg_max_shared(int64_t* agg, const int64_t val) {
  atomicMax64(agg, val);
}

extern "C"
__device__ void agg_max_double_shared(int64_t* agg, const double val) {
  atomicMax(reinterpret_cast<double*>(agg), val);
}

extern "C"
__device__ void agg_min_shared(int64_t* agg, const int64_t val) {
  atomicMin64(agg, val);
}

extern "C"
__device__ void agg_min_double_shared(int64_t* agg, const double val) {
  atomicMin(reinterpret_cast<double*>(agg), val);
}

extern "C"
__device__ void agg_id_shared(int64_t* agg, const int64_t val) {
  *agg = val;
}

extern "C"
__device__ void agg_id_double_shared(int64_t* agg, const double val) {
  *agg = *(reinterpret_cast<const int64_t*>(&val));
}

#define DEF_SKIP_AGG(base_agg_func)                                                                         \
extern "C"                                                                                                  \
__device__ void base_agg_func##_skip_val_shared(int64_t* agg, const int64_t val, const int64_t skip_val) {  \
  if (val != skip_val) {                                                                                    \
    base_agg_func##_shared(agg, val);                                                                       \
  }                                                                                                         \
}

DEF_SKIP_AGG(agg_count)
DEF_SKIP_AGG(agg_sum)

__device__ int64_t atomicMin64SkipVal(int64_t* address, int64_t val, const int64_t skip_val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        assumed == skip_val ? val : min((long long) val, (long long) assumed));
    } while (assumed != old);

    return old;
}

extern "C"
__device__ void agg_min_skip_val_shared(int64_t* agg, const int64_t val, const int64_t skip_val) {
  if (val != skip_val) {
    atomicMin64SkipVal(agg, val, skip_val);
  }
}

__device__ int64_t atomicMax64SkipVal(int64_t* address, int64_t val, const int64_t skip_val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        assumed == skip_val ? val : max((long long) val, (long long) assumed));
    } while (assumed != old);

    return old;
}

extern "C"
__device__ void agg_max_skip_val_shared(int64_t* agg, const int64_t val, const int64_t skip_val) {
  if (val != skip_val) {
    atomicMax64SkipVal(agg, val, skip_val);
  }
}

#undef DEF_SKIP_AGG

#define DEF_SKIP_AGG(base_agg_func)                                                                       \
extern "C"                                                                                                \
__device__ void base_agg_func##_skip_val_shared(int64_t* agg, const double val, const double skip_val) {  \
  if (val != skip_val) {                                                                                  \
    base_agg_func##_shared(agg, val);                                                                     \
  }                                                                                                       \
}

DEF_SKIP_AGG(agg_count_double)
DEF_SKIP_AGG(agg_sum_double)

__device__ double atomicMinDblSkipVal(double* address, double val, const double skip_val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int skip_val_as_ull = *((unsigned long long int*) &skip_val);
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed == skip_val_as_ull
          ? *((unsigned long long int*) &val) : __double_as_longlong(min(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

extern "C"
__device__ void agg_min_double_skip_val_shared(int64_t* agg, const double val, const double skip_val) {
  if (val != skip_val) {
    atomicMinDblSkipVal(reinterpret_cast<double*>(agg), val, skip_val);
  }
}

__device__ double atomicMaxDblSkipVal(double* address, double val, const double skip_val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int skip_val_as_ull = *((unsigned long long int*) &skip_val);
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed == skip_val_as_ull
          ? *((unsigned long long int*) &val) : __double_as_longlong(max(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

extern "C"
__device__ void agg_max_double_skip_val_shared(int64_t* agg, const double val, const double skip_val) {
  if (val != skip_val) {
    atomicMaxDblSkipVal(reinterpret_cast<double*>(agg), val, skip_val);
  }
}

#undef DEF_SKIP_AGG

#define SECSPERMIN	60L
#define MINSPERHOUR	60L
#define HOURSPERDAY	24L
#define SECSPERHOUR	(SECSPERMIN * MINSPERHOUR)
#define SECSPERDAY	(SECSPERHOUR * HOURSPERDAY)
#define DAYSPERWEEK	7
#define MONSPERYEAR	12

#define YEAR_BASE	1900

/* move epoch from 01.01.1970 to 01.03.2000 - this is the first day of new
 * 400-year long cycle, right after additional day of leap year. This adjustment
 * is required only for date calculation, so instead of modifying time_t value
 * (which would require 64-bit operations to work correctly) it's enough to
 * adjust the calculated number of days since epoch. */
#define EPOCH_ADJUSTMENT_DAYS	11017
/* year to which the adjustment was made */
#define ADJUSTED_EPOCH_YEAR	2000
/* 1st March of 2000 is Wednesday */
#define ADJUSTED_EPOCH_WDAY	3
/* there are 97 leap years in 400-year periods. ((400 - 97) * 365 + 97 * 366) */
#define DAYS_PER_400_YEARS	146097L
/* there are 24 leap years in 100-year periods. ((100 - 24) * 365 + 24 * 366) */
#define DAYS_PER_100_YEARS	36524L
/* there is one leap year every 4 years */
#define DAYS_PER_4_YEARS	(3 * 365 + 366)
/* number of days in a non-leap year */
#define DAYS_PER_YEAR		365
/* number of days in January */
#define DAYS_IN_JANUARY		31
/* number of days in non-leap February */
#define DAYS_IN_FEBRUARY	28

extern "C"
__device__  tm* gmtime_r_cuda(const time_t *tim_p, tm* res) {
  const int month_lengths[2][MONSPERYEAR] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
  };
  long days, rem;
  const time_t lcltime = *tim_p;
  int year, month, yearday, weekday;
  int years400, years100, years4, remainingyears;
  int yearleap;
  const int *ip;

  days = ((long)lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
  rem = ((long)lcltime) % SECSPERDAY;
  if (rem < 0)
    {
      rem += SECSPERDAY;
      --days;
    }

  /* compute hour, min, and sec */
  res->tm_hour = (int) (rem / SECSPERHOUR);
  rem %= SECSPERHOUR;
  res->tm_min = (int) (rem / SECSPERMIN);
  res->tm_sec = (int) (rem % SECSPERMIN);

  /* compute day of week */
  if ((weekday = ((ADJUSTED_EPOCH_WDAY + days) % DAYSPERWEEK)) < 0)
    weekday += DAYSPERWEEK;
  res->tm_wday = weekday;

  /* compute year & day of year */
  years400 = days / DAYS_PER_400_YEARS;
  days -= years400 * DAYS_PER_400_YEARS;
  /* simplify by making the values positive */
  if (days < 0)
    {
      days += DAYS_PER_400_YEARS;
      --years400;
    }

  years100 = days / DAYS_PER_100_YEARS;
  if (years100 == 4) /* required for proper day of year calculation */
    --years100;
  days -= years100 * DAYS_PER_100_YEARS;
  years4 = days / DAYS_PER_4_YEARS;
  days -= years4 * DAYS_PER_4_YEARS;
  remainingyears = days / DAYS_PER_YEAR;
  if (remainingyears == 4) /* required for proper day of year calculation */
    --remainingyears;
  days -= remainingyears * DAYS_PER_YEAR;

  year = ADJUSTED_EPOCH_YEAR + years400 * 400 + years100 * 100 + years4 * 4 +
      remainingyears;

  /* If remainingyears is zero, it means that the years were completely
   * "consumed" by modulo calculations by 400, 100 and 4, so the year is:
   * 1. a multiple of 4, but not a multiple of 100 or 400 - it's a leap year,
   * 2. a multiple of 4 and 100, but not a multiple of 400 - it's not a leap
   * year,
   * 3. a multiple of 4, 100 and 400 - it's a leap year.
   * If years4 is non-zero, it means that the year is not a multiple of 100 or
   * 400 (case 1), so it's a leap year. If years100 is zero (and years4 is zero
   * - due to short-circuiting), it means that the year is a multiple of 400
   * (case 3), so it's also a leap year. */
  yearleap = remainingyears == 0 && (years4 != 0 || years100 == 0);

  /* adjust back to 1st January */
  yearday = days + DAYS_IN_JANUARY + DAYS_IN_FEBRUARY + yearleap;
  if (yearday >= DAYS_PER_YEAR + yearleap)
    {
      yearday -= DAYS_PER_YEAR + yearleap;
      ++year;
    }
  res->tm_yday = yearday;
  res->tm_year = year - YEAR_BASE;

  /* Because "days" is the number of days since 1st March, the additional leap
   * day (29th of February) is the last possible day, so it doesn't matter much
   * whether the year is actually leap or not. */
  ip = month_lengths[1];
  month = 2;
  while (days >= ip[month])
    {
      days -= ip[month];
      if (++month >= MONSPERYEAR)
        month = 0;
    }
  res->tm_mon = month;
  res->tm_mday = days + 1;

  res->tm_isdst = 0;

  return (res);
}

#include "ExtractFromTime.cpp"
#include "../Utils/ChunkIter.cpp"
#include "../Utils/StringLike.cpp"

extern "C"
__device__ uint64_t string_decode(int8_t* chunk_iter_, int64_t pos) {
  // TODO(alex): de-dup, the x64 version is basically identical
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  VarlenDatum vd;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, pos, false, &vd, &is_end);
  return vd.is_null
    ? 0
    : (reinterpret_cast<uint64_t>(vd.pointer) & 0xffffffffffff) | (static_cast<uint64_t>(vd.length) << 48);
}

extern "C"
__device__ int32_t merge_error_code(const int32_t err_code, int32_t* merged_err_code) {
  if (err_code) {
    int32_t assumed = *merged_err_code;
    int32_t old;
    do {
      old = atomicCAS(merged_err_code, assumed, err_code);
    } while (old != assumed);
  }
  __syncthreads();
  return *merged_err_code;
}
