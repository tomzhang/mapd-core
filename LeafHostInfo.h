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

enum class NodeRole { DbLeaf, String };

class LeafHostInfo {
 public:
  LeafHostInfo(const std::string& host, const uint16_t port, const NodeRole role)
      : host_(host), port_(port), role_(role) {}

  const std::string& getHost() const { return host_; }

  uint16_t getPort() const { return port_; }

  NodeRole getRole() const { return role_; }

  static std::vector<LeafHostInfo> parseClusterConfig(const std::string& file_path);

 private:
  std::string host_;
  uint16_t port_;
  NodeRole role_;
};

#endif  // LEAFHOSTINFO_H
