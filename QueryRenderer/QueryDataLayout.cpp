#include "QueryDataLayout.h"
#include <Rendering/Renderer/GL/Resources/GLShaderBlockLayout.h>

namespace QueryRenderer {

using Rendering::GL::Resources::GLBufferLayoutShPtr;
using Rendering::GL::Resources::GLInterleavedBufferLayout;
using Rendering::GL::Resources::GLSequentialBufferLayout;
using Rendering::GL::Resources::GLBufferAttrType;
using Rendering::GL::Resources::GLShaderBlockLayout;
using Rendering::GL::Resources::GLShaderBlockLayoutShPtr;

const std::string QueryDataLayout::dummyPrefix = "_dummy";

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
  if (!_convertedLayout) {
    int dummyCnt = 0;
    switch (layoutType) {
      case LayoutType::INTERLEAVED: {
        // TODO(croot): make a base interleaved/sequential buffer class
        // perhaps called BaseIntSeqBufferLayout
        // so we don't have to duplicate code here.
        // And support adding the attrnames and types in a constructor of
        // that base class?
        _convertedLayout.reset(new GLInterleavedBufferLayout());
        GLInterleavedBufferLayout* layout = dynamic_cast<GLInterleavedBufferLayout*>(_convertedLayout.get());
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
      } break;
      case LayoutType::SEQUENTIAL: {
        _convertedLayout.reset(new GLSequentialBufferLayout());
        GLSequentialBufferLayout* layout = dynamic_cast<GLSequentialBufferLayout*>(_convertedLayout.get());
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
      } break;
      default:
        CHECK(false);
        break;
    }
  }

  return _convertedLayout;
}

GLShaderBlockLayoutShPtr QueryDataLayout::convertToUniformBufferLayout() {
  if (!_convertedLayout) {
    int dummyCnt = 0;
    _convertedLayout.reset(new GLShaderBlockLayout());
    GLShaderBlockLayout* layout = dynamic_cast<GLShaderBlockLayout*>(_convertedLayout.get());
    layout->beginAddingAttrs();
    for (size_t i = 0; i < attrNames.size(); ++i) {
      switch (attrTypes[i]) {
        case AttrType::UINT:
          layout->addAttribute<unsigned int>(attrNames[i]);
          break;
        case AttrType::INT:
          layout->addAttribute<int>(attrNames[i]);
          break;
        case AttrType::FLOAT:
          layout->addAttribute<float>(attrNames[i]);
          break;
        case AttrType::DOUBLE:
          layout->addAttribute<double>(attrNames[i]);
          break;
        case AttrType::UINT64:
          // TODO(croot): support 64-bit ints
          // So for the time being, add a dummy attr for the first
          // 32bits of the attr, and then the real attr for the
          // last 32bits.
          layout->addAttribute<unsigned int>(attrNames[i]);
          layout->addAttribute<unsigned int>(dummyPrefix + std::to_string(dummyCnt++));
          break;
        case AttrType::INT64:
          // TODO(croot): support 64-bit ints (see UINT64)
          layout->addAttribute<int>(attrNames[i]);
          layout->addAttribute<int>(dummyPrefix + std::to_string(dummyCnt++));
          break;
      }
    }
    layout->endAddingAttrs();
  } else {
    RUNTIME_EX_ASSERT(dynamic_cast<GLShaderBlockLayout*>(_convertedLayout.get()),
                      "Cannot convert query data layout to a uniform buffer layout. It has already been converted to a "
                      "vertex buffer layout");
  }

  return std::dynamic_pointer_cast<GLShaderBlockLayout>(_convertedLayout);
}

}  // namespace QueryRenderer
