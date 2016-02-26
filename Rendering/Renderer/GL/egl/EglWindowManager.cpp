#include "EglWindowManager.h"
#include "EglGLWindow.h"
#include "EglGLRenderer.h"

namespace Rendering {
namespace GL {
namespace EGL {

EglWindowManagerImpl::EglWindowManagerImpl() {
  // #ifdef MAPDGL_MULTITHREADED
  //   XInitThreads();
  // #endif
}

EglWindowManagerImpl::~EglWindowManagerImpl() {
}

std::string EglWindowManagerImpl::getName() const {
  return "EGL";
}

size_t EglWindowManagerImpl::getNumGpus() const {
  return _displayMgr.getNumGpus();
}

size_t EglWindowManagerImpl::getNumGpus(const Settings::WindowSettings& settings) const {
  return getNumGpus();
}

WindowShPtr EglWindowManagerImpl::createWindow(const Settings::WindowSettings& settings) {
  // TODO(croot): Do I need to deal with the possibility of multi threading?
  // Do I need to XLockDisplay/XUnlockDisplay? This may need to be taken
  // care of in the window class.

  WindowShPtr windowPtr(new EglGLWindow(settings, _displayMgr));
  return windowPtr;
}

RendererShPtr EglWindowManagerImpl::createRenderer(const Settings::RendererSettings& settings) {
  RendererShPtr rendererPtr(new EglGLRenderer(settings));
  return rendererPtr;
}

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering
