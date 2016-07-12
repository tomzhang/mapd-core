#ifndef RENDERING_GL_RESOURCES_GLVERTEXBUFFER_H_
#define RENDERING_GL_RESOURCES_GLVERTEXBUFFER_H_

#include "Types.h"
#include "GLBufferLayout.h"
#include "GLBaseBuffer.h"
#include "GLVertexArray.h"
#include "../TypeGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLVertexBuffer : public GLBaseBuffer {
 public:
  ~GLVertexBuffer();

  bool hasAttribute(const std::string& attrName) const;

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const;

  GLBufferAttrType getAttributeType(const std::string& attrName) const;

  int numAttributes() const;
  int numItems() const;
  int numVertices() const;

  const GLBufferAttrInfo& operator[](size_t i) const;

  GLBufferLayoutShPtr getBufferLayout() const { return _layoutPtr; }
  void setBufferLayout(const GLBufferLayoutShPtr& layoutPtr, size_t numItems);

  void bindToShader(GLShader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "");

  void bufferData(void* data, size_t numItems, size_t numBytesPerItem);

  void debugPrintData(void* data, size_t idx);

 private:
  explicit GLVertexBuffer(const RendererWkPtr& rendererPtr,
                          BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                          BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);
  explicit GLVertexBuffer(const RendererWkPtr& rendererPtr,
                          size_t numBytes,
                          BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                          BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLVertexBuffer(const RendererWkPtr& rendererPtr,
                          const GLBufferLayoutShPtr& layoutPtr,
                          BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                          BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  template <typename T>
  explicit GLVertexBuffer(const RendererWkPtr& rendererPtr,
                          const std::vector<T>& data,
                          const GLBufferLayoutShPtr& layoutPtr,
                          BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                          BufferAccessFreq accessFreq = BufferAccessFreq::STATIC)
      : GLVertexBuffer(rendererPtr, layoutPtr, accessType, accessFreq) {
    // TODO(croot): validate that the data and the layout align
    // NOTE: _size will be set in the bufferData() call
    bufferData((void*)&data[0], data.size(), sizeof(T));
  }

  void _makeEmpty() final;

  void _bindToShaderInternal(GLShader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "");

  void _addVertexArray(GLVertexArrayShPtr& vao);
  void _deleteVertexArray(GLVertexArray* vao);
  void _deleteAllVertexArrays();
  void _updateVertexArrays();

  GLBufferLayoutShPtr _layoutPtr;
  size_t _numItems;
  std::set<GLVertexArrayWkPtr, std::owner_less<GLVertexArrayWkPtr>> _vaoRefs;

  friend class ::Rendering::GL::GLResourceManager;
  friend class ::Rendering::GL::Resources::GLVertexArray;
  // friend void ::Rendering::GL::Resources::GLVertexArray::initialize(const VboAttrToShaderAttrMap&, GLRenderer*);
  // friend void ::Rendering::GL::Resources::GLVertexArray::_deleteAllVertexBuffers();
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLVERTEXBUFFER_H_
