#ifndef RENDERING_GL_RESOURCES_GLUNIFORMBUFFER_H_
#define RENDERING_GL_RESOURCES_GLUNIFORMBUFFER_H_

#include "Types.h"
#include "GLShaderBlockLayout.h"
#include "GLBaseBuffer.h"
#include "../TypeGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLUniformBuffer : public GLBaseBuffer {
 public:
  ~GLUniformBuffer();

  GLResourceType getResourceType() const final { return GLResourceType::UNIFORMBUFFER; }

  bool hasAttribute(const std::string& attrName) const;

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const;

  GLBufferAttrType getAttributeType(const std::string& attrName) const;

  int numAttributes() const;
  size_t numItems() const;
  size_t getNumBytesInBlock() const;
  size_t getNumBytesPerItem() const;
  size_t getNumAlignmentBytes() const;

  const GLBufferAttrInfo& operator[](size_t i) const;

  GLShaderBlockLayoutShPtr getBufferLayout() const { return _shaderBlockLayoutPtr; }
  void setBufferLayout(const GLShaderBlockLayoutShPtr& shaderBlockLayoutPtr, size_t numItems);

  // void bindToShader(GLShader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "");

  void bufferData(void* data, size_t numItems, size_t numBytesPerItem);

  template <typename T>
  void setAttrData(size_t idx, const std::string& attrName, T attrValue) {
    _bufferSubData(attrName, &attrValue, sizeof(T), 1, idx);
  }

  template <typename T>
  void setAttrData(size_t idx, const std::string& attrName, const std::vector<T>& attrValue) {
    _bufferSubData(attrName, &attrValue[0], sizeof(T), attrValue.size(), idx);
  }

  template <typename T, size_t N>
  void setAttrData(size_t idx, const std::string& attrName, const std::array<T, N>& attrValue) {
    _bufferSubData(attrName, &attrValue[0], sizeof(T), N, idx);
  }

 private:
  explicit GLUniformBuffer(const RendererWkPtr& rendererPtr,
                           BufferAccessType accessType = BufferAccessType::WRITE,
                           BufferAccessFreq accessFreq = BufferAccessFreq::DYNAMIC);

  explicit GLUniformBuffer(const RendererWkPtr& rendererPtr,
                           size_t numBytes,
                           BufferAccessType accessType = BufferAccessType::WRITE,
                           BufferAccessFreq accessFreq = BufferAccessFreq::DYNAMIC);

  explicit GLUniformBuffer(const RendererWkPtr& rendererPtr,
                           const GLShaderBlockLayoutShPtr& shaderBlockLayoutPtr,
                           size_t numItems = 0,
                           BufferAccessType accessType = BufferAccessType::WRITE,
                           BufferAccessFreq accessFreq = BufferAccessFreq::DYNAMIC);

  template <typename T>
  explicit GLUniformBuffer(const RendererWkPtr& rendererPtr,
                           const std::vector<T>& data,
                           const GLShaderBlockLayoutShPtr& shaderBlockLayoutPtr,
                           BufferAccessType accessType = BufferAccessType::WRITE,
                           BufferAccessFreq accessFreq = BufferAccessFreq::DYNAMIC)
      : GLUniformBuffer(rendererPtr, shaderBlockLayoutPtr, 0, accessType, accessFreq) {
    // TODO(croot): validate that the data and the layout align
    // NOTE: _size will be set in the bufferData() call
    bufferData((void*)&data[0], data.size(), sizeof(T));
  }

  void _makeEmpty() final;

  void _bufferSubData(const std::string& attrName,
                      const void* data,
                      size_t bytesPerComponent,
                      size_t numComponents,
                      int idx);

  // void _bindToShaderInternal(GLShader* activeShader, const std::string& attr = "", const std::string& shaderAttr =
  // "");

  GLShaderBlockLayoutShPtr _shaderBlockLayoutPtr;
  size_t _numItems;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLUNIFORMBUFFER_H_
