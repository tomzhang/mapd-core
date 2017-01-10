#ifndef RENDERING_GL_RESOURCES_GLSHADERBLOCKLAYOUT_H_
#define RENDERING_GL_RESOURCES_GLSHADERBLOCKLAYOUT_H_

#include "GLBufferLayout.h"
#include "GLShader.h"
#include "Enums.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLShaderBlockLayout : public GLBaseBufferLayout {
 public:
  GLShaderBlockLayout(const std::set<std::string>& supportedExtensions,
                      ShaderBlockLayoutType layoutType = ShaderBlockLayoutType::STD140);
  ~GLShaderBlockLayout();

  bool operator==(const GLShaderBlockLayout& layout) const;
  bool operator!=(const GLShaderBlockLayout& layout) const;

  ShaderBlockLayoutType getLayoutType() const { return _layoutType; }
  size_t getNumBytesInBlock() const {
    CHECK(!_addingAttrs)
        << "Please call the endAddingAttrs() method before calling methods requiring the all attributes to be added.";
    return _itemByteSize;
  }

  size_t getNumBytesPerItem() const final { return getNumAlignmentBytes(); }

  void beginAddingAttrs();
  void endAddingAttrs();

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName) {
    CHECK(_addingAttrs) << "Please call the \"beginAddingAttrs()\" method before adding attributes.";
    RUNTIME_EX_ASSERT(
        _layoutType == ShaderBlockLayoutType::STD140 || _layoutType == ShaderBlockLayoutType::STD430,
        "Cannot add attribute " + attrName +
            ". Attributes can only be added to a std140 or std430 shader block layout via this public interface.");

    RUNTIME_EX_ASSERT(!hasAttribute(attrName) && attrName.length(),
                      "GLShaderBlockLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

    int offset = 0;
    GLBufferAttrType type = getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);
    BaseTypeGL* typeGL = attrTypeInfo[enumVal].get();

    // TODO(croot): there's a lot to do to make this fully functional,
    // i.e. supporting matrices and structs, and arrays of all types
    // Starting with the basics.

    int dataSz = sizeof(T);
    switch (_layoutType) {
      case ShaderBlockLayoutType::STD140:
        // the rules for how memory is laid out for these types are defined
        // in section 7.6.2.2. (page 137) of the latest opengl spec:
        // https://www.opengl.org/registry/doc/glspec45.core.pdf#page=159

        if (numComponents == 3) {
          offset = dataSz * 4;
        } else {
          offset = dataSz * numComponents;
        }
        break;
      case ShaderBlockLayoutType::STD430:
        THROW_RUNTIME_EX("Layout type std430 has yet to be implemented.")
        break;
      default:
        break;
    }

    _attrMap.push_back(BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, typeGL, -1, _itemByteSize)));
    _itemByteSize += offset;
  }

  // TODO(croot): what about other shader block options (i.e. bindings/locations, matrix storage order, etc.)
  std::string buildShaderBlockCode(const std::string& blockName,
                                   const std::string& instanceName = "",
                                   StorageQualifier storageQualifier = StorageQualifier::IN);

  static size_t getNumAlignmentBytes();

 private:
  GLShaderBlockLayout(const std::set<std::string>& supportedExtensions,
                      ShaderBlockLayoutType layoutType,
                      const GLShaderShPtr& shaderPtr,
                      size_t blockByteSize);

  // TODO(croot): if these values can vary per-gpu, then we probably
  // need to store these on a per-object basis or query directly from
  // the context when needed. But if these are defined per driver, then
  // keeping them static is fine.
  static int uniformBufferOffsetAlignment;
  static int maxUniformBlockSize;

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName, GLuint attrIdx) {
    CHECK(_addingAttrs) << "Please call the \"beginAddingAttrs()\" method before adding attributes.";
    RUNTIME_EX_ASSERT(!hasAttribute(attrName) && attrName.length(),
                      "GLShaderBlockLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

    CHECK(attrIdx == _attrMap.size()) << "Trying to add an attribute " << attrName
                                      << " to GLShaderBlockLayout, but it's being added out of order. It's index is "
                                      << attrIdx << " but the attr with index " << _attrMap.size()
                                      << " is next in line to be added.";

    CHECK(_layoutType == ShaderBlockLayoutType::PACKED || _layoutType == ShaderBlockLayoutType::SHARED);

    GLint offset = -1;

    // TODO(croot): does the appropriate context need to be active?
    GLShaderShPtr shader = _shaderPtr.lock();

    CHECK(shader && _itemByteSize > 0);
    MAPD_CHECK_GL_ERROR(glGetActiveUniformsiv(shader->getId(), 1, &attrIdx, GL_UNIFORM_OFFSET, &offset));
    CHECK(offset >= 0) << "Attribute " << attrName
                       << " is not a part of a shader storage block. Cannot retrieve its offset";

    GLBufferAttrType type = getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);
    _attrMap.push_back(
        BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), -1, offset)));
  }

  void bindToShader(GLShader* activeShader,
                    const size_t usedBytes,
                    const size_t offsetBytes,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "") const final;

  static void _initNumAlignmentBytes();
  static void _initMaxUniformBlockSize();

  GLShaderWkPtr _shaderPtr;
  ShaderBlockLayoutType _layoutType;
  bool _addingAttrs;

  friend class ::Rendering::GL::Resources::GLShader;
  friend struct ::Rendering::GL::Resources::detail::UniformBlockAttrInfo;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLSHADERBLOCKLAYOUT_H_
