#ifndef QUERY_DATA_LAYOUT_H_
#define QUERY_DATA_LAYOUT_H_

#include "QueryRendererError.h"
#include "BufferLayout.h"

#include <unordered_map>
#include <vector>

namespace MapD_Renderer {

struct QueryDataLayout {
  enum class AttrType { UINT = 0, INT, FLOAT, DOUBLE, UINT64, INT64 };
  enum class LayoutType { INTERLEAVED = 0, SEQUENTIAL };

  size_t numRows;

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

  QueryDataLayout(const size_t numRows,
                  const std::vector<std::string>& attrNames,
                  const std::vector<AttrType>& attrTypes,
                  const std::unordered_map<std::string, std::string>& attrAliasToName,
                  const size_t numKeys = 0,  // TODO(croot) - fill out the number of keys still, all would be irrelevant
                                             // except the first key, which would be the one to check the invalid key
                                             // against.
                  const int64_t invalidKey = std::numeric_limits<int64_t>::max(),
                  const LayoutType layoutType = LayoutType::INTERLEAVED)
      : numRows(numRows),
        numKeys(numKeys),
        invalidKey(invalidKey),
        attrNames(attrNames),
        attrTypes(attrTypes),
        attrAliasToName(attrAliasToName),
        layoutType(layoutType) {
    RUNTIME_EX_ASSERT(
        attrNames.size() == attrTypes.size(),
        "QueryDataLayout constructor: The number of attribute names must match the number of attribute types.");
  }

  BufferLayoutShPtr convertToBufferLayout() {
    static const std::string dummyPrefix = "___dummy___";
    int dummyCnt = 0;
    switch (layoutType) {
      case LayoutType::INTERLEAVED: {
        // TODO(croot): make a base interleaved/sequential buffer class
        // perhaps called BaseIntSeqBufferLayout
        // so we don't have to duplicate code here.
        // And support adding the attrnames and types in a constructor of
        // that base class?
        InterleavedBufferLayout* layout = new InterleavedBufferLayout();
        for (size_t i = 0; i < attrNames.size(); ++i) {
          switch (attrTypes[i]) {
            case AttrType::UINT:
              layout->addAttribute(attrNames[i], BufferAttrType::UINT);
              break;
            case AttrType::INT:
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              break;
            case AttrType::FLOAT:
              layout->addAttribute(attrNames[i], BufferAttrType::FLOAT);
              break;
            case AttrType::DOUBLE:
              layout->addAttribute(attrNames[i], BufferAttrType::DOUBLE);
              break;
            case AttrType::UINT64:
              // TODO(croot): support 64-bit ints
              // So for the time being, add a dummy attr for the first
              // 32bits of the attr, and then the real attr for the
              // last 32bits.
              layout->addAttribute(attrNames[i], BufferAttrType::UINT);
              layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), BufferAttrType::UINT);
              break;
            case AttrType::INT64:
              // TODO(croot): support 64-bit ints (see UINT64)
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), BufferAttrType::INT);
              break;
          }
        }
        return BufferLayoutShPtr(layout);
      }
      case LayoutType::SEQUENTIAL: {
        SequentialBufferLayout* layout = new SequentialBufferLayout();
        for (size_t i = 0; i < attrNames.size(); ++i) {
          switch (attrTypes[i]) {
            case AttrType::UINT:
              layout->addAttribute(attrNames[i], BufferAttrType::UINT);
              break;
            case AttrType::INT:
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              break;
            case AttrType::FLOAT:
              layout->addAttribute(attrNames[i], BufferAttrType::FLOAT);
              break;
            case AttrType::DOUBLE:
              layout->addAttribute(attrNames[i], BufferAttrType::DOUBLE);
              break;
            case AttrType::UINT64:
              // TODO(croot): support 64-bit ints
              // So for the time being, add a dummy attr for the first
              // 32bits of the attr, and then the real attr for the
              // last 32bits.
              layout->addAttribute(attrNames[i], BufferAttrType::UINT);
              layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), BufferAttrType::UINT);
              break;
            case AttrType::INT64:
              // TODO(croot): support 64-bit ints
              // So for the time being, add a dummy attr for the first
              // 32bits of the attr, and then the real attr for the
              // last 32bits.
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), BufferAttrType::INT);
              break;
          }
        }
        return BufferLayoutShPtr(layout);
      }
      default:
        CHECK(false);
        return nullptr;
    }
  }
};

}  // namespace MapD_Renderer

#endif  // QUERY_DATA_LAYOUT_H_
