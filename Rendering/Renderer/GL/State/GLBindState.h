#ifndef RENDERING_GL_STATE_GLBINDSTATE_H_
#define RENDERING_GL_STATE_GLBINDSTATE_H_

#include "../Resources/Types.h"
#include "../Resources/Enums.h"
#include "../Types.h"
#include "GL/glew.h"

namespace Rendering {
namespace GL {
namespace State {

class GLBindState {
 public:
  Resources::GLTexture2dWkPtr boundTex2d;
  Resources::GLTexture2dArrayWkPtr boundTex2dArray;
  Resources::GLVertexBufferWkPtr boundVbo;
  Resources::GLFramebufferWkPtr boundReadFbo;
  Resources::GLFramebufferWkPtr boundDrawFbo;
  Resources::GLShaderWkPtr boundShader;
  Resources::GLVertexArrayWkPtr boundVao;
  Resources::GLRenderbufferWkPtr boundRbo;
  Resources::GLPixelBuffer2dWkPtr boundReadPbo;
  Resources::GLPixelBuffer2dWkPtr boundWritePbo;
  Resources::GLUniformBufferWkPtr boundUbo;
  Resources::GLIndexBufferWkPtr boundIbo;
  Resources::GLIndirectDrawBufferWkPtr boundIndirectBuffer;

  ~GLBindState();

  void bindResource(const Resources::GLResourceShPtr& rsrc);

  void bindTexture2d(const Resources::GLTexture2dShPtr& texRsrc);
  void bindTexture2dArray(const Resources::GLTexture2dArrayShPtr& texArrayRsrc);
  void bindVertexBuffer(const Resources::GLVertexBufferShPtr& vboRsrc);
  void bindFramebuffer(Resources::FboBind bindType, const Resources::GLFramebufferShPtr& fbRsrc);
  void bindShader(const Resources::GLShaderShPtr& shaderRsrc);
  void bindVertexArray(const Resources::GLVertexArrayShPtr& vaoRsrc);
  void bindRenderbuffer(const Resources::GLRenderbufferShPtr& rboRsrc);
  void bindReadPixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc);
  void bindWritePixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc);
  void bindUniformBuffer(const Resources::GLUniformBufferShPtr& uboRsrc);
  void bindIndexBuffer(const Resources::GLIndexBufferShPtr& iboRsrc);
  void bindIndirectDrawBuffer(const Resources::GLIndirectDrawBufferShPtr& indirectRsrc);

  Resources::GLTexture2dShPtr getBoundTexture2d() const;
  bool hasBoundTexture2d() const;

  Resources::GLTexture2dArrayShPtr getBoundTexture2dArray() const;
  bool hasBoundTexture2dArray() const;

  Resources::GLVertexBufferShPtr getBoundVertexBuffer() const;
  bool hasBoundVertexBuffer() const;

  Resources::GLFramebufferShPtr getBoundFramebuffer(const Resources::FboBind bindType) const;
  bool hasBoundFramebuffer(const Resources::FboBind bindType) const;

  Resources::GLShaderShPtr getBoundShader() const;
  bool hasBoundShader() const;

  Resources::GLVertexArrayShPtr getBoundVertexArray() const;
  bool hasBoundVertexArray() const;

  Resources::GLRenderbufferShPtr getBoundRenderbuffer() const;
  bool hasBoundRenderbuffer() const;

  Resources::GLPixelBuffer2dShPtr getBoundReadPixelBuffer() const;
  bool hasBoundReadPixelBuffer() const;

  Resources::GLPixelBuffer2dShPtr getBoundWritePixelBuffer() const;
  bool hasBoundWritePixelBuffer() const;

  Resources::GLUniformBufferShPtr getBoundUniformBuffer() const;
  bool hasBoundUniformBuffer() const;

  Resources::GLIndexBufferShPtr getBoundIndexBuffer() const;
  bool hasBoundIndexBuffer() const;

  Resources::GLIndirectDrawBufferShPtr getBoundIndirectDrawBuffer() const;
  bool hasBoundIndirectDrawBuffer() const;

  Resources::GLIndirectDrawVertexBufferShPtr getBoundIndirectDrawVertexBuffer() const;
  bool hasBoundIndirectDrawVertexBuffer() const;

  Resources::GLIndirectDrawIndexBufferShPtr getBoundIndirectDrawIndexBuffer() const;
  bool hasBoundIndirectDrawIndexBuffer() const;

 private:
  GLBindState(GLRenderer* prntRenderer);

  GLRenderer* _prntRenderer;

  friend class ::Rendering::GL::GLRenderer;
};

}  // namespace State
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_STATE_GLBINDSTATE_H_
