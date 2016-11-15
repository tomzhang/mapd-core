/*
 * @file    LeafHostInfo.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Information about leaf nodes and utilities to parse a cluster configuration file.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef LEAFHOSTINFO_H
#define LEAFHOSTINFO_H

#include <string>
#include <vector>

class LeafHostInfo {
 public:
  LeafHostInfo(const std::string& host, const uint16_t port);

  const std::string& getHost() const;

  uint16_t getPort() const;

  static std::vector<LeafHostInfo> parseClusterConfig(const std::string& file_path);

 private:
  std::string host_;
  uint16_t port_;
};

#endif  // LEAFHOSTINFO_H
