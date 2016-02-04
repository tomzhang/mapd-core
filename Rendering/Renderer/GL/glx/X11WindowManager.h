#ifndef RENDERING_GL_GLX_X11WINDOWMANAGER_H_
#define RENDERING_GL_GLX_X11WINDOWMANAGER_H_

#include "../../../WindowManager.h"
#include "X11DisplayManager.h"
#include <string>

namespace Rendering {
namespace GL {

namespace GLX {

/**
 * An abstract interface for windowing toolkits.
 *
 * This class is intended to be overwritten per system windowing library
 * to initialize the windowing system and create system-specific windows.
 *
 */
class X11WindowManagerImpl : public BaseWindowManagerImpl {
 public:
  ~X11WindowManagerImpl();

  std::string getName() const final;
  size_t getNumGpus() const final;
  size_t getNumGpus(const Settings::WindowSettings& settings) const final;

 protected:
  X11WindowManagerImpl();

  WindowShPtr createWindow(const Settings::WindowSettings& settings) final;
  RendererShPtr createRenderer(const Settings::RendererSettings& settings) final;

 private:
  X11DisplayManager _displayMgr;

  friend class ::Rendering::WindowManager;
};

}  // namespace GLX

}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLX_X11WINDOWMANAGER_H_
