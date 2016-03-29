#ifndef RENDERING_GL_RESOURCES_GLSHADERBLOCKLAYOUT_H_
#define RENDERING_GL_RESOURCES_GLSHADERBLOCKLAYOUT_H_

#include "GLBufferLayout.h"
#include "GLShader.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLShaderBlockLayout : public GLBaseBufferLayout {
 public:
  // see: https://www.opengl.org/wiki/Interface_Block_(GLSL)#Memory_layout
  // for a description of the different layout types
  enum class LayoutType { PACKED = 0, SHARED, STD140, STD430 };

  GLShaderBlockLayout(const GLShaderShPtr& shaderPtr, size_t blockByteSize, LayoutType layoutType = LayoutType::SHARED);
  ~GLShaderBlockLayout();

  LayoutType getLayoutType() const { return _layoutType; }
  size_t getNumBytesInBlock() const { return _vertexByteSize; }

  static size_t getNumAlignmentBytes();

 private:
  // TODO(croot): if these values can vary per-gpu, then we probably
  // need to store these on a per-object basis or query directly from
  // the context when needed. But if these are defined per driver, then
  // keeping them static is fine.
  static int uniformBufferOffsetAlignment;
  static int maxUniformBlockSize;

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName, GLuint attrIdx) {
    RUNTIME_EX_ASSERT(!hasAttribute(attrName) && attrName.length(),
                      "GLShaderBlockLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

    CHECK(attrIdx == _attrMap.size()) << "Trying to add an attribute " << attrName
                                      << " to GLShaderBlockLayout, but it's being added out of order. It's index is "
                                      << attrIdx << " but the attr with index " << _attrMap.size()
                                      << " is next in line to be added.";

    GLint offset = 0;

    switch (_layoutType) {
      case LayoutType::PACKED:
      case LayoutType::SHARED: {
        GLShaderShPtr shader = _shaderPtr.lock();
        CHECK(shader != nullptr);
        MAPD_CHECK_GL_ERROR(glGetActiveUniformsiv(shader->getId(), 1, &attrIdx, GL_UNIFORM_OFFSET, &offset));
        CHECK(offset >= 0) << "Attribute " << attrName
                           << " is not a part of a shader storage block. Cannot retrieve its offset";
        break;
      }

      case LayoutType::STD140:
        THROW_RUNTIME_EX("Layout type std140 has yet to be implemented.")
        break;
      case LayoutType::STD430:
        THROW_RUNTIME_EX("Layout type std430 has yet to be implemented.")
        break;
      default:
        THROW_RUNTIME_EX("Unsupported shader storage block layout type: " +
                         std::to_string(static_cast<int>(_layoutType)));
        break;
    }

    GLBufferAttrType type = detail::getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);
    _attrMap.push_back(
        BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[enumVal].get(), -1, offset)));
  }

  void bindToShader(GLShader* activeShader,
                    int numActiveBufferItems,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "");

  static void _initNumAlignmentBytes();
  static void _initMaxUniformBlockSize();

  GLShaderWkPtr _shaderPtr;
  LayoutType _layoutType;

  friend class ::Rendering::GL::Resources::GLShader;
  friend struct ::Rendering::GL::Resources::detail::UniformBlockAttrInfo;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLSHADERBLOCKLAYOUT_H_
