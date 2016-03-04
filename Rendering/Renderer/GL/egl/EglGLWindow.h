#ifndef RENDERING_GL_EGL_EGLGLWINDOW_H_
#define RENDERING_GL_EGL_EGLGLWINDOW_H_

#include "../GLWindow.h"
#include "Types.h"

namespace Rendering {
namespace GL {
namespace EGL {

class EglGLWindow : public GLWindow {
 public:
  ~EglGLWindow();

  size_t getGpuId() const final;

  EGLSurface getEGLSurface() const { return _surface; }
  EglDisplayShPtr getEGLDisplayPtr() const { return _dpyPtr; }
  EGLConfig getEGLConfig() const { return _eglCfg; }

  void swapBuffers() final;

 private:
  explicit EglGLWindow(const Settings::WindowSettings& windowSettings,
                       EglDisplayManager& displayMgr,
                       const WindowShPtr& prnt = nullptr);

  void _initEGLDisplay(EglDisplayManager& displayMgr);

  void _init(Renderer* renderer);
  EGLConfig _chooseEGLConfig(EglGLRenderer* renderer);
  void _initSurface(EglGLRenderer* renderer, EGLConfig& cfg);
  void _initRenderer(EglGLRenderer* renderer);
  void _initSwap(EglGLRenderer* renderer);

  EglDisplayShPtr _dpyPtr;
  int _gpuId;
  EGLSurface _surface;
  EGLConfig _eglCfg;

  friend class EglWindowManagerImpl;
};

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_EGL_EGLGLWINDOW_H_
