#include "QueryDataLayout.h"

namespace QueryRenderer {

using Rendering::GL::Resources::GLBufferLayoutShPtr;
using Rendering::GL::Resources::GLInterleavedBufferLayout;
using Rendering::GL::Resources::GLSequentialBufferLayout;
using Rendering::GL::Resources::GLBufferAttrType;

QueryDataLayout::QueryDataLayout(const std::vector<std::string>& attrNames,
                                 const std::vector<AttrType>& attrTypes,
                                 const std::unordered_map<std::string, std::string>& attrAliasToName,
                                 const size_t numKeys,
                                 const int64_t invalidKey,
                                 const LayoutType layoutType)
    : numKeys(numKeys),
      invalidKey(invalidKey),
      attrNames(attrNames),
      attrTypes(attrTypes),
      attrAliasToName(attrAliasToName),
      layoutType(layoutType) {
  RUNTIME_EX_ASSERT(
      attrNames.size() == attrTypes.size(),
      "QueryDataLayout constructor: The number of attribute names must match the number of attribute types.");
}

GLBufferLayoutShPtr QueryDataLayout::convertToBufferLayout() {
  static const std::string dummyPrefix = "___dummy___";
  int dummyCnt = 0;
  switch (layoutType) {
    case LayoutType::INTERLEAVED: {
      // TODO(croot): make a base interleaved/sequential buffer class
      // perhaps called BaseIntSeqBufferLayout
      // so we don't have to duplicate code here.
      // And support adding the attrnames and types in a constructor of
      // that base class?
      GLBufferLayoutShPtr rtn(new GLInterleavedBufferLayout());
      GLInterleavedBufferLayout* layout = dynamic_cast<GLInterleavedBufferLayout*>(rtn.get());
      for (size_t i = 0; i < attrNames.size(); ++i) {
        switch (attrTypes[i]) {
          case AttrType::UINT:
            layout->addAttribute(attrNames[i], GLBufferAttrType::UINT);
            break;
          case AttrType::INT:
            layout->addAttribute(attrNames[i], GLBufferAttrType::INT);
            break;
          case AttrType::FLOAT:
            layout->addAttribute(attrNames[i], GLBufferAttrType::FLOAT);
            break;
          case AttrType::DOUBLE:
            layout->addAttribute(attrNames[i], GLBufferAttrType::DOUBLE);
            break;
          case AttrType::UINT64:
            // TODO(croot): support 64-bit ints
            // So for the time being, add a dummy attr for the first
            // 32bits of the attr, and then the real attr for the
            // last 32bits.
            layout->addAttribute(attrNames[i], GLBufferAttrType::UINT);
            layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::UINT);
            break;
          case AttrType::INT64:
            // TODO(croot): support 64-bit ints (see UINT64)
            layout->addAttribute(attrNames[i], GLBufferAttrType::INT);
            layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::INT);
            break;
        }
      }
      return rtn;
    }
    case LayoutType::SEQUENTIAL: {
      GLBufferLayoutShPtr rtn(new GLSequentialBufferLayout());
      GLSequentialBufferLayout* layout = dynamic_cast<GLSequentialBufferLayout*>(rtn.get());
      for (size_t i = 0; i < attrNames.size(); ++i) {
        switch (attrTypes[i]) {
          case AttrType::UINT:
            layout->addAttribute(attrNames[i], GLBufferAttrType::UINT);
            break;
          case AttrType::INT:
            layout->addAttribute(attrNames[i], GLBufferAttrType::INT);
            break;
          case AttrType::FLOAT:
            layout->addAttribute(attrNames[i], GLBufferAttrType::FLOAT);
            break;
          case AttrType::DOUBLE:
            layout->addAttribute(attrNames[i], GLBufferAttrType::DOUBLE);
            break;
          case AttrType::UINT64:
            // TODO(croot): support 64-bit ints
            // So for the time being, add a dummy attr for the first
            // 32bits of the attr, and then the real attr for the
            // last 32bits.
            layout->addAttribute(attrNames[i], GLBufferAttrType::UINT);
            layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::UINT);
            break;
          case AttrType::INT64:
            // TODO(croot): support 64-bit ints
            // So for the time being, add a dummy attr for the first
            // 32bits of the attr, and then the real attr for the
            // last 32bits.
            layout->addAttribute(attrNames[i], GLBufferAttrType::INT);
            layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::INT);
            break;
        }
      }
      return rtn;
    }
    default:
      CHECK(false);
      return nullptr;
  }
}

}  // namespace QueryRenderer
