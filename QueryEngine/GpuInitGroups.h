/*
 * @file    GpuInitGroups.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef GPUINITGROUPS_H
#define GPUINITGROUPS_H

#include <stdint.h>

void init_group_by_buffer_on_device(int64_t* groups_buffer,
                                    const int64_t* init_vals,
                                    const uint32_t groups_buffer_entry_count,
                                    const uint32_t key_qw_count,
                                    const uint32_t agg_col_count,
                                    const bool keyless,
                                    const int8_t warp_size,
                                    const size_t block_size_x,
                                    const size_t grid_size_x);

void init_columnar_group_by_buffer_on_device(int64_t* groups_buffer,
                                             const int64_t* init_vals,
                                             const uint32_t groups_buffer_entry_count,
                                             const uint32_t key_qw_count,
                                             const uint32_t agg_col_count,
                                             const bool keyless,
                                             const size_t block_size_x,
                                             const size_t grid_size_x);

#endif  // GPUINITGROUPS_H
