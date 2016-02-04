#ifndef RENDERING_GL_GLRENDERER_H_
#define RENDERING_GL_GLRENDERER_H_

#include "Types.h"
#include "Resources/Types.h"
#include "../../Renderer.h"
#include "../../Objects/Types.h"
#include "State/GLBindState.h"
#include <GL/glew.h>
#include <cstdint>
#include <vector>

namespace Rendering {

namespace GL {

class GLRenderer : public Renderer {
 public:
  virtual ~GLRenderer();

  void initialize() final;
  virtual void makeActiveOnCurrentThread(const Window* window = nullptr) override;
  virtual void makeInactive() override;
  bool isActiveOnCurrentThread() final;

  // TODO(croot): add a renderer type enum and get the renderer type?
  bool isGLRenderer() final { return true; }
  GLResourceManagerShPtr getResourceManager();

#ifdef GLEW_MX
  GLEWContext* glewGetContext();
  const GLEWContext* glewGetContext() const;
#endif

  /** State related functions **/

  // TODO(croot): Eventually put in a state machine class
  // that could handle push/pop of state for passes
  // in addition to querying, but for now, just to get things
  // up and running, just do a fixed-function state lookup
  // bool getBoolAttr(GLenum attrName);
  // int getIntAttr(GLenum attrName);
  // float getFloatAttr(GLenum attrName);
  // double getDoubleAttr(GLenum attrName);
  // int64_t getInt64Attr(GLenum attrName);

  template <typename T>
  T getAttr(GLenum attrName);

  template <typename T>
  std::vector<T> getAttrv(GLenum attrName);

  void setClearColor(const Objects::ColorRGBA& color) final;
  void setClearColor(float r, float g, float b, float a) final;

  void clearColor() final;
  void clearAll() final;

  void setViewport(int x, int y, int width, int height);
  void setViewport(const Objects::Viewport& viewport);

  void enable(GLenum attr);
  void disable(GLenum attr);

  void setBlendFunc(GLenum srcFactor, GLenum dstFactor);

  void drawVertexBuffers(GLenum primitiveMode, int startIndex, int numItemsToDraw);

  void bindResource(const Resources::GLResourceShPtr& rsrc);
  void bindFramebuffer(Resources::FboBind bindType, const Resources::GLFramebufferShPtr& fboRsrc);
  void bindShader(const Resources::GLShaderShPtr& shaderRsrc);
  void bindVertexArray(const Resources::GLVertexArrayShPtr& vaoRsrc);

  Resources::GLFramebufferShPtr getBoundFbo(Resources::FboBind bindType) const;
  bool hasBoundFbo(Resources::FboBind bindType) const;

  Resources::GLShaderShPtr getBoundShader() const;
  bool hasBoundShader() const;

  Resources::GLVertexArrayShPtr getBoundVertexArray() const;
  bool hasBoundVertexArray() const;

  static const GLRenderer* getCurrentThreadRenderer();

 protected:
  // GLRenderer(const WindowShPtr& parentWindowPtr);
  GLRenderer(const RendererSettings& settings);

  void _initGLEW(const GLWindow* primaryWindow = nullptr);
  void _cleanupResources();

  GLResourceManagerShPtr _glRsrcMgrPtr;

 private:
  virtual void initializeGL() = 0;

  void setResourceManager(const GLResourceManagerShPtr& glRsrcMgr);

  bool _glewInitialized;

#ifdef GLEW_MX
  /** Needed for multiple rendering contexts using GLEW. **/
  // TODO(croot): in http://glew.sourceforge.net/advanced.html
  // where it discusses GLEW MX, item #4 says:
  //
  // "Make sure that you call glewInit() after creating the GLEWContext object "
  // "in each rendering context. Note, that the GLEWContext pointer returned by "
  // "glewGetContext() has to reside in global or thread-local memory."
  //
  // I'm not sure if creating the context here as part of a GLRenderer class
  // satisfies this point.. However, pointers to GLRenderer instances can be put
  // in thread local memory.
  GLEWContext _glewContext;
#endif

  State::GLBindState _bindState;

  friend class GLWindow;
  friend class ::Rendering::WindowManager;
};

// template <int size = 1>
// std::array<bool, size> GLRenderer::getAttrv(GLenum attrName) {
// }

}  // namespace GL

}  // namespace Rendering

#endif  // RENDERING_GL_GLRENDERER_H_
