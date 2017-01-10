#ifndef QUERYRENDERER_QUERYDATALAYOUT_H_
#define QUERYRENDERER_QUERYDATALAYOUT_H_

#include "Types.h"
#include <Rendering/Renderer/GL/Resources/Enums.h>
#include <Rendering/Renderer/GL/Resources/GLBufferLayout.h>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>

namespace QueryRenderer {

struct QueryDataLayout {
  enum class AttrType { UINT = 0, INT, FLOAT, DOUBLE, UINT64, INT64 };
  enum class LayoutType { INTERLEAVED = 0, SEQUENTIAL };

  // TODO(croot): add size_t numKeys --- each row can have
  // multiple keys. This value would indicate how many.
  // We can default it to use one, and the invalid key
  // would only match against the very first key. None of the
  // others would concern me.
  size_t numKeys;
  int64_t invalidKey;
  std::vector<std::string> attrNames;
  std::vector<AttrType> attrTypes;
  std::unordered_map<std::string, std::string> attrAliasToName;
  LayoutType layoutType;

  QueryDataLayout(const std::vector<std::string>& attrNames,
                  const std::vector<AttrType>& attrTypes,
                  const std::unordered_map<std::string, std::string>& attrAliasToName,
                  const size_t numKeys = 0,  // TODO(croot) - fill out the number of keys still, all would be irrelevant
                                             // except the first key, which would be the one to check the invalid key
                                             // against.
                  const int64_t invalidKey = std::numeric_limits<int64_t>::max(),
                  const LayoutType layoutType = LayoutType::INTERLEAVED);

  ~QueryDataLayout() {}

  Rendering::GL::Resources::GLBufferLayoutShPtr convertToBufferLayout(const std::set<std::string>& supportedExtensions);
  Rendering::GL::Resources::GLShaderBlockLayoutShPtr convertToUniformBufferLayout(
      const std::set<std::string>& supportedExtensions);

  bool hasBeenConverted() const { return _convertedLayout != nullptr; }
  Rendering::GL::Resources::GLBufferLayoutShPtr getBufferLayout() const { return _convertedLayout; }

  bool operator==(const QueryDataLayout& layout) const;
  bool operator!=(const QueryDataLayout& layout) const { return !operator==(layout); }

 private:
  static const std::string dummyPrefix;

  ::Rendering::GL::Resources::GLBufferLayoutShPtr _convertedLayout;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYDATALAYOUT_H_
