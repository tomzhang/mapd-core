#include "X11WindowManager.h"
// #include "../Settings.h"
#include "GlxGLWindow.h"
#include "GlxGLRenderer.h"
#include "X11DisplayManager.h"
#include <X11/Xresource.h>

namespace Rendering {
namespace GL {
namespace GLX {

X11WindowManagerImpl::X11WindowManagerImpl() : _displayMgr() {
  // #ifdef MAPDGL_MULTITHREADED
  //   XInitThreads();
  // #endif

  XInitThreads();
}

X11WindowManagerImpl::~X11WindowManagerImpl() {
}

std::string X11WindowManagerImpl::getName() const {
  return "X11/GLX";
}

size_t X11WindowManagerImpl::getNumGpus() const {
  return _displayMgr.getNumGpus();
}

size_t X11WindowManagerImpl::getNumGpus(const Settings::WindowSettings& settings) const {
  std::string displayStr = settings.getStrSetting(Settings::StrSetting::DISPLAY);
  return _displayMgr.getNumGpus(displayStr);
}

WindowShPtr X11WindowManagerImpl::createWindow(const WindowSettings& settings) {
  // TODO(croot): Do I need to deal with the possibility of multi threading?
  // Do I need to XLockDisplay/XUnlockDisplay? This may need to be taken
  // care of in the window class.

  WindowShPtr windowPtr(new GlxGLWindow(settings, _displayMgr));
  return windowPtr;
}

RendererShPtr X11WindowManagerImpl::createRenderer(const RendererSettings& settings) {
  RendererShPtr rendererPtr(new GlxGLRenderer(settings));
  return rendererPtr;
}

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering
