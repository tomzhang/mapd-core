#include "backendrendererSetup.h"

__global__ void setup_kernel_gpu_wrapper(curandState* state) {
  int idx = threadIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void get_random_data_gpu_wrapper(curandState* state, Row* row, int numPts) {
  int idx = threadIdx.x;

  curandState localState = state[idx];

  // write output vertex
  // pos[idx] = make_double4(curand_uniform(&localState),
  //                         curand_uniform(&localState),
  //                         curand_uniform(&localState),
  //                         floor(curand_uniform(&localState) * 2.9999) + 1.0);

  row[idx].key = idx;
  row[idx].x = curand_uniform(&localState);
  row[idx].y = curand_uniform(&localState);
  row[idx].val = curand_uniform(&localState);
  row[idx].party = int64_t(floor(curand_uniform(&localState) * 2.9999) + 1.0);

  state[idx] = localState;
}

__global__ void get_random_numbers_gpu_wrapper(curandState* state, float* results) {
  int idx = threadIdx.x;
  curandState localState = state[idx];

  results[idx] = curand_uniform(&localState);

  // copy local state back to global state
  state[idx] = localState;
}

void setup_kernel(curandState* state, const size_t block_size_x, const size_t grid_size_x) {
  setup_kernel_gpu_wrapper << <grid_size_x, block_size_x>>> (state);
}

void get_random_data(curandState* state, Row* row, int numPts, const size_t block_size_x, const size_t grid_size_x) {
  get_random_data_gpu_wrapper << <grid_size_x, block_size_x>>> (state, row, numPts);
}

void get_random_numbers(curandState* state, float* results, const size_t block_size_x, const size_t grid_size_x) {
  get_random_numbers_gpu_wrapper << <grid_size_x, block_size_x>>> (state, results);
}
