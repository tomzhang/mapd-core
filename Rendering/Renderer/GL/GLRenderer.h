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
#include <mutex>

namespace Rendering {

namespace GL {

class GLRenderer : public Renderer {
 public:
  virtual ~GLRenderer();

  void initialize() final;
  void makeActiveOnCurrentThread(Window* window = nullptr) final;
  void makeInactive() final;
  bool isActiveOnCurrentThread(Window* window = nullptr) final;

  // TODO(croot): add a renderer type enum and get the renderer type?
  bool isGLRenderer() final { return true; }
  GLResourceManagerShPtr getResourceManager();

#ifdef GLEW_MX
  GLEWContext* glewGetContext();
  const GLEWContext* glewGetContext() const;
#endif

  const GLWindow* getPrimaryGLWindow() const;
  GLWindow* getPrimaryGLWindow();
  size_t getGpuId() const;

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
  Objects::Viewport getViewport() const;

  void enable(GLenum attr);
  void disable(GLenum attr);

  void setBlendFunc(GLenum srcFactor, GLenum dstFactor, int drawBufferId = -1);
  void setBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha, int drawBufferId = -1);

  void setBlendEquation(GLenum mode, int drawBufferId = -1);
  void setBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha, int drawBufferId = -1);

  void setMinSampleShading(const float minSampleShading);

  void bindResource(const Resources::GLResourceShPtr& rsrc);
  void bindTexture2d(const Resources::GLTexture2dShPtr& texRsrc);
  void bindVertexBuffer(const Resources::GLVertexBufferShPtr& vboRsrc);
  void bindFramebuffer(Resources::FboBind bindType, const Resources::GLFramebufferShPtr& fboRsrc);
  void bindShader(const Resources::GLShaderShPtr& shaderRsrc);
  void bindVertexArray(const Resources::GLVertexArrayShPtr& vaoRsrc);
  void bindRenderbuffer(const Resources::GLRenderbufferShPtr& rboRsrc);
  void bindReadPixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc);
  void bindWritePixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc);
  void bindUniformBuffer(const Resources::GLUniformBufferShPtr& uboRsrc);
  void bindIndexBuffer(const Resources::GLIndexBufferShPtr& uboRsrc);
  void bindIndirectDrawBuffer(const Resources::GLIndirectDrawBufferShPtr& indirectRsrc);

  Resources::GLTexture2dShPtr getBoundTexture2d() const;
  bool hasBoundTexture2d() const;

  Resources::GLVertexBufferShPtr getBoundVertexBuffer() const;
  bool hasBoundVertexBuffer() const;

  Resources::GLFramebufferShPtr getBoundFramebuffer(Resources::FboBind bindType) const;
  bool hasBoundFramebuffer(Resources::FboBind bindType) const;

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

  void drawVertexBuffers(GLenum primitiveMode, size_t startIndex = 0, int numItemsToDraw = -1);
  void drawIndirectVertexBuffers(GLenum primitiveMode, size_t startIndex = 0, int numItemsToDraw = -1);

  void drawIndexBuffers(GLenum primitiveMode, size_t startIndex = 0, int numItemsToDraw = -1);
  void drawIndirectIndexBuffers(GLenum primitiveMode, size_t startIndex = 0, int numItemsToDraw = -1);

  void getReadFramebufferPixels(GLenum attachment,
                                size_t startx,
                                size_t starty,
                                size_t width,
                                size_t height,
                                GLenum format,
                                GLenum type,
                                GLvoid* data);

  void getBoundTexture2dPixels(size_t width, size_t height, GLenum format, GLenum type, GLvoid* data);

  static GLRenderer* getCurrentThreadRenderer();
  static Window* getCurrentThreadWindow();
  // static void setInactiveRendererCurrentOnThread();

 protected:
  // GLRenderer(const WindowShPtr& parentWindowPtr);
  GLRenderer(const RendererSettings& settings);

  virtual void _makeActiveOnCurrentThreadImpl(Window* window) = 0;
  virtual void _makeInactiveImpl() = 0;

  void _initGLEW(GLWindow* primaryWindow = nullptr);
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

  static std::mutex _currRendererMtx;

  friend class GLWindow;
  friend class ::Rendering::WindowManager;
};

// template <int size = 1>
// std::array<bool, size> GLRenderer::getAttrv(GLenum attrName) {
// }

}  // namespace GL

}  // namespace Rendering

#endif  // RENDERING_GL_GLRENDERER_H_
