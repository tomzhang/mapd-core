#ifndef RENDERING_GL_EGL_EGLWINDOWMANAGER_H_
#define RENDERING_GL_EGL_EGLWINDOWMANAGER_H_

#include "../../../WindowManager.h"
#include "EglDisplayManager.h"

namespace Rendering {
namespace GL {
namespace EGL {

class EglWindowManagerImpl : public BaseWindowManagerImpl {
 public:
  ~EglWindowManagerImpl();

 protected:
  EglWindowManagerImpl();

  std::string getName() const final;
  size_t getNumGpus() const final;
  size_t getNumGpus(const Settings::WindowSettings& settings) const final;

  WindowShPtr createWindow(const Settings::WindowSettings& settings) final;
  RendererShPtr createRenderer(const Settings::RendererSettings& settings) final;

 private:
  EglDisplayManager _displayMgr;

  friend class ::Rendering::WindowManager;
};

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_EGL_EGLWINDOWMANAGER_H_
