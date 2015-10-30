#ifndef BACKENDRENDERER_SETUP_H_
#define BACKENDRENDERER_SETUP_H_

#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
// #include "QueryRenderManager.h"

typedef struct {
  int64_t key;
  double x;
  double y;
  double val;
  int64_t party;

  // static ::MapD_Renderer::QueryDataLayout getQueryDataLayout(size_t numRows) {
  //   return ::MapD_Renderer::QueryDataLayout(numRows,
  //                                           {"x", "y", "val", "party"},
  //                                           {::MapD_Renderer::QueryDataLayout::AttrType::DOUBLE,
  //                                            ::MapD_Renderer::QueryDataLayout::AttrType::DOUBLE,
  //                                            ::MapD_Renderer::QueryDataLayout::AttrType::DOUBLE,
  //                                            ::MapD_Renderer::QueryDataLayout::AttrType::INT64});
  // }
} Row;

void setup_kernel(curandState* state, const size_t block_size_x, const size_t grid_size_x);
void get_random_data(curandState* state, Row* row, int numPts, const size_t block_size_x, const size_t grid_size_x);
void get_random_numbers(curandState* state, float* results, const size_t block_size_x, const size_t grid_size_x);

#endif  // BACKENDRENDERER_SETUP_H_
