#ifndef RENDERING_GL_RESOURCES_GLVERTEXBUFFER_H_
#define RENDERING_GL_RESOURCES_GLVERTEXBUFFER_H_

#include "Types.h"
#include "GLBufferLayout.h"
#include "GLLayoutBuffer.h"
#include "GLVertexArrayResource.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLVertexBuffer : public GLVertexArrayResource<const GLBufferLayoutShPtr&>,
                       public GLLayoutBuffer<GLBaseBufferLayout> {
 public:
  ~GLVertexBuffer();

  size_t numVertices(const GLBufferLayoutShPtr& layoutPtr = nullptr) const;

  void bufferData(void* data, const size_t numBytes, const GLBufferLayoutShPtr& layoutPtr = nullptr) final;

  void debugPrintData(void* data, const GLBufferLayoutShPtr& layoutPtr, size_t idx);

 private:
  explicit GLVertexBuffer(const RendererWkPtr& rendererPtr,
                          BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                          BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLVertexBuffer(const RendererWkPtr& rendererPtr,
                          size_t numBytes,
                          BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                          BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  void _makeEmpty() final;

  void _bindToShaderInternal(GLShader* activeShader,
                             const GLBufferLayoutShPtr& layoutPtr = nullptr,
                             const std::string& attr = "",
                             const std::string& shaderAttr = "");

  void _layoutAddedCB(const BoundLayoutData& layoutData) final {}
  void _layoutUpdatedCB(const BoundLayoutData& preUpdateLayoutData, const BoundLayoutData& postUpdateLayoutData) final {
    _markVertexArraysDirty(preUpdateLayoutData.layoutPtr);
  }
  void _layoutReplacedCB(const BoundLayoutData& replacedLayoutData,
                         const BoundLayoutData& replacementLayoutData) final {
    _markVertexArraysDirty(replacedLayoutData.layoutPtr);
  }
  void _layoutDeletedCB(const BoundLayoutData& layoutData) final { _markVertexArraysDirty(layoutData.layoutPtr); }
  void _allLayoutsDeletedCB() final { _markVertexArraysDirty(nullptr); }

  bool _doesVaoUseThisResource(const GLVertexArrayShPtr& vao, const GLBufferLayoutShPtr& layout) final;
  void _setVaoDirtyFlag(GLVertexArrayShPtr& vao, const bool dirtyFlag) final;

  friend class ::Rendering::GL::GLResourceManager;
  friend class ::Rendering::GL::Resources::GLVertexArray;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLVERTEXBUFFER_H_
