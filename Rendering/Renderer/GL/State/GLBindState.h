#ifndef RENDERING_GL_STATE_GLBINDSTATE_H_
#define RENDERING_GL_STATE_GLBINDSTATE_H_

#include "../Resources/Types.h"
#include "../Types.h"
#include "GL/glew.h"

namespace Rendering {
namespace GL {
namespace State {

class GLBindState {
 public:
  Resources::GLVertexBufferWkPtr boundVbo;
  Resources::GLFramebufferWkPtr boundReadFbo;
  Resources::GLFramebufferWkPtr boundDrawFbo;

  Resources::GLShaderWkPtr boundShader;

  Resources::GLVertexArrayWkPtr boundVao;

  ~GLBindState();

  void bindResource(const Resources::GLResourceShPtr& rsrc);

  void bindVertexBuffer(const Resources::GLVertexBufferShPtr& vboRsrc);
  void bindFramebuffer(Resources::FboBind bindType, const Resources::GLFramebufferShPtr& fbRsrc);
  void bindShader(const Resources::GLShaderShPtr& shaderRsrc);
  void bindVertexArray(const Resources::GLVertexArrayShPtr& vaoRsrc);

  Resources::GLVertexBufferShPtr getBoundVbo() const;
  bool hasBoundVbo() const;

  Resources::GLFramebufferShPtr getBoundFbo(const Resources::FboBind bindType) const;
  bool hasBoundFbo(const Resources::FboBind bindType) const;

  Resources::GLShaderShPtr getBoundShader() const { return boundShader.lock(); }
  bool hasBoundShader() const { return boundShader.lock() != nullptr; }

  Resources::GLVertexArrayShPtr getBoundVertexArray() const { return boundVao.lock(); }
  bool hasBoundVertexArray() const { return boundVao.lock() != nullptr; }

 private:
  GLBindState(GLRenderer* prntRenderer);

  GLRenderer* _prntRenderer;

  friend class ::Rendering::GL::GLRenderer;
};

}  // namespace State
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_STATE_GLBINDSTATE_H_
