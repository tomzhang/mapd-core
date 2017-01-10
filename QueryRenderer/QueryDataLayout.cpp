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

GLBufferLayoutShPtr QueryDataLayout::convertToBufferLayout(const std::set<std::string>& supportedExtensions) {
  if (!_convertedLayout) {
    int dummyCnt = 0;
    bool supportsInt64 = supportedExtensions.find("GL_NV_vertex_attrib_integer_64bit") != supportedExtensions.end();
    switch (layoutType) {
      case LayoutType::INTERLEAVED: {
        // TODO(croot): make a base interleaved/sequential buffer class
        // perhaps called BaseIntSeqBufferLayout
        // so we don't have to duplicate code here.
        // And support adding the attrnames and types in a constructor of
        // that base class?
        _convertedLayout.reset(new GLInterleavedBufferLayout(supportedExtensions));
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
              if (supportsInt64) {
                layout->addAttribute(attrNames[i], GLBufferAttrType::UINT64);
              } else {
                // 64-bit ints are not supported, so add a dummy attr for the
                // first 32 bits and then the real attr for the remaining
                // 32 bits
                layout->addAttribute(attrNames[i], GLBufferAttrType::UINT);
                layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::UINT);
              }
              break;
            case AttrType::INT64:
              if (supportsInt64) {
                layout->addAttribute(attrNames[i], GLBufferAttrType::INT64);
              } else {
                // 64-bit ints are not supported, so add a dummy attr for the
                // first 32 bits and then the real attr for the remaining
                // 32 bits
                layout->addAttribute(attrNames[i], GLBufferAttrType::INT);
                layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::INT);
              }

              break;
          }
        }
      } break;
      case LayoutType::SEQUENTIAL: {
        _convertedLayout.reset(new GLSequentialBufferLayout(supportedExtensions));
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
              if (supportsInt64) {
                layout->addAttribute(attrNames[i], GLBufferAttrType::UINT64);
              } else {
                // 64-bit ints are not supported, so add a dummy attr for the
                // first 32 bits and then the real attr for the remaining
                // 32 bits
                layout->addAttribute(attrNames[i], GLBufferAttrType::UINT);
                layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::UINT);
              }
              break;
            case AttrType::INT64:
              if (supportsInt64) {
                layout->addAttribute(attrNames[i], GLBufferAttrType::INT64);
              } else {
                // 64-bit ints are not supported, so add a dummy attr for the
                // first 32 bits and then the real attr for the remaining
                // 32 bits
                layout->addAttribute(attrNames[i], GLBufferAttrType::INT);
                layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), GLBufferAttrType::INT);
              }
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

GLShaderBlockLayoutShPtr QueryDataLayout::convertToUniformBufferLayout(
    const std::set<std::string>& supportedExtensions) {
  if (!_convertedLayout) {
    int dummyCnt = 0;
    bool supportsInt64 = supportedExtensions.find("GL_NV_vertex_attrib_integer_64bit") != supportedExtensions.end();
    _convertedLayout.reset(new GLShaderBlockLayout(supportedExtensions));
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
          if (supportsInt64) {
            layout->addAttribute<uint64_t>(attrNames[i]);
          } else {
            // 64-bit ints are not supported, so add a dummy attr for the
            // first 32 bits and then the real attr for the remaining
            // 32 bits
            layout->addAttribute<unsigned int>(attrNames[i]);
            layout->addAttribute<unsigned int>(dummyPrefix + std::to_string(dummyCnt++));
          }
          break;
        case AttrType::INT64:
          if (supportsInt64) {
            layout->addAttribute<int64_t>(attrNames[i]);
          } else {
            // 64-bit ints are not supported, so add a dummy attr for the
            // first 32 bits and then the real attr for the remaining
            // 32 bits
            layout->addAttribute<int>(attrNames[i]);
            layout->addAttribute<int>(dummyPrefix + std::to_string(dummyCnt++));
          }
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

bool QueryDataLayout::operator==(const QueryDataLayout& layout) const {
  if (hasBeenConverted() && layout.hasBeenConverted()) {
    return *_convertedLayout == *(layout._convertedLayout);
  } else {
    return attrTypes == layout.attrTypes;
  }
}

}  // namespace QueryRenderer
