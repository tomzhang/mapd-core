#ifndef RENDERING_GL_EGL_EGLGLRENDERER_H_
#define RENDERING_GL_EGL_EGLGLRENDERER_H_

#include "../GLRenderer.h"
#include "Types.h"

namespace Rendering {
namespace GL {
namespace EGL {

class EglGLRenderer : public GLRenderer {
 public:
  ~EglGLRenderer();

  EglDisplayShPtr getEGLDisplayPtr() const { return _dpyPtr; }
  EGLContext getEGLContext() const { return _eglCtx; }

 private:
  EglGLRenderer(const RendererSettings& settings);

  /** initialize function called by parent class **/
  void initializeGL() final;

  void _initEGLDisplay(const EglGLWindow* primaryWindow);
  EGLConfig _chooseEGLConfig(const EglGLWindow* primaryWindow);

  bool _verifyWindowIsAttachable(const Window* window) final;

  void _makeActiveOnCurrentThreadImpl(Window* window) final;
  void _makeInactiveImpl() final;

  EglDisplayShPtr _dpyPtr;
  EGLContext _eglCtx;

  friend class EglWindowManagerImpl;
};

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_EGL_EGLGLRENDERER_H_
