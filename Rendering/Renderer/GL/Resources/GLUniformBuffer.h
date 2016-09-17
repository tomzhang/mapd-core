#ifndef RENDERING_GL_RESOURCES_GLUNIFORMBUFFER_H_
#define RENDERING_GL_RESOURCES_GLUNIFORMBUFFER_H_

#include "Types.h"
#include "GLShaderBlockLayout.h"
#include "GLLayoutBuffer.h"
#include "../TypeGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLUniformBuffer : public GLLayoutBuffer<GLShaderBlockLayout> {
 public:
  ~GLUniformBuffer();

  size_t getNumBytesInBlock(const GLShaderBlockLayoutShPtr& layoutPtr = nullptr) const;
  size_t getNumBytesPerItem(const GLShaderBlockLayoutShPtr& layoutPtr = nullptr) const;
  size_t getNumAlignmentBytes() const;

  void bufferData(void* data, const size_t numBytes, const GLShaderBlockLayoutShPtr& layoutPtr = nullptr) final;

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

  void _makeEmpty() final;

  void _validateBufferLayout(size_t numBytes,
                             size_t offsetBytes,
                             const GLBufferLayoutShPtr& layoutPtr,
                             bool replaceExistingLayout = false,
                             const std::string& errPrefix = "") final;

  void _bufferSubData(const std::string& attrName,
                      const void* data,
                      size_t bytesPerComponent,
                      size_t numComponents,
                      int idx);

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLUNIFORMBUFFER_H_
