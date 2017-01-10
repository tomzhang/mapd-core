#include "LeafHostInfo.h"
#include "QueryEngine/JsonAccessors.h"
#include <rapidjson/filereadstream.h>
#include <cstdio>

namespace {

class ReadOnlyFileStream {
 public:
  ReadOnlyFileStream(const std::string& file_path) {
    file_stream_ = fopen(file_path.c_str(), "r");
    CHECK(file_stream_);
  }

  ~ReadOnlyFileStream() { fclose(file_stream_); }

  std::FILE* getStream() const { return file_stream_; }

 private:
  std::FILE* file_stream_;
};

NodeRole parse_role(const std::string& str) {
  if (str == "dbleaf") {
    return NodeRole::DbLeaf;
  }
  CHECK_EQ(std::string("string"), str);
  return NodeRole::String;
}

}  // namespace

std::vector<LeafHostInfo> LeafHostInfo::parseClusterConfig(const std::string& file_path) {
  ReadOnlyFileStream file_stream(file_path);
  char read_buffer[65536];
  rapidjson::FileReadStream json_stream(file_stream.getStream(), read_buffer, sizeof(read_buffer));
  rapidjson::Document cluster_json;
  cluster_json.ParseStream(json_stream);
  CHECK(cluster_json.IsArray());
  std::vector<LeafHostInfo> leaves;
  for (auto leaf_it = cluster_json.Begin(); leaf_it != cluster_json.End(); ++leaf_it) {
    const auto& leaf = *leaf_it;
    CHECK(leaf.IsObject());
    const auto host = json_str(field(leaf, "host"));
    const auto port = static_cast<uint16_t>(json_i64(field(leaf, "port")));
    const auto role = parse_role(json_str(field(leaf, "role")));
    leaves.emplace_back(host, port, role);
  }
  return leaves;
}
