#include "WindowManager.h"
#include "Window.h"
#include "Renderer.h"
#include "Settings/WindowSettings.h"
#include "Settings/RendererSettings.h"

// #include "Settings/RendererSettings.h"

#ifdef MAPDGL
#include "Renderer/GL/GLRenderer.h"
#include "Renderer/GL/GLResourceManager.h"
#endif

#ifdef MAPDGL_GLX
#include "Renderer/GL/glx/X11WindowManager.h"
#elif MAPDGL_EGL
#include "Renderer/GL/egl/EglWindowManager.h"
#endif  // MAPDGL_GLX

#include <iostream>

namespace Rendering {

using Settings::WindowSettings;
using Settings::RendererSettings;

// store the main thread id for comparison as WindowManager() instances
// must be created on the main thread
const std::thread::id WindowManager::_mainThread = std::this_thread::get_id();

WindowManager::WindowManager() : _implPtr(nullptr) {
  if (std::this_thread::get_id() != _mainThread) {
    THROW_RUNTIME_EX("A WindowManager instance can only be created on the main thread");
  }

#ifdef MAPDGL_GLX
  _implPtr.reset(new Rendering::GL::GLX::X11WindowManagerImpl());
#elif MAPDGL_EGL
  _implPtr.reset(new Rendering::GL::EGL::EglWindowManagerImpl());
#endif  // MAPDGL_GLX
}

WindowManager::~WindowManager() {
  std::cerr << "CROOT - Window Manager destructor" << std::endl;
}

size_t WindowManager::getNumGpus() const {
  CHECK(_implPtr != nullptr);
  return _implPtr->getNumGpus();
}

size_t WindowManager::getNumGpus(const Settings::WindowSettings& settings) const {
  CHECK(_implPtr != nullptr);
  return _implPtr->getNumGpus(settings);
}

WindowShPtr WindowManager::createWindow(const WindowSettings& settings) {
  CHECK(_implPtr != nullptr);
  return _implPtr->createWindow(settings);
}

RendererShPtr WindowManager::createRenderer(const RendererSettings& settings) {
  CHECK(_implPtr != nullptr);
  RendererShPtr renderer = _implPtr->createRenderer(settings);

#ifdef MAPDGL
  if (renderer->isGLRenderer()) {
    // TODO(croot): consider making a createGLRenderer function instead.
    // That might make more sense.
    // In that case we'd know for sure we've built a GL renderer

    GL::GLRenderer* glRenderer = dynamic_cast<GL::GLRenderer*>(renderer.get());
    GL::GLResourceManagerShPtr rsrcMgr(new GL::GLResourceManager(renderer));
    glRenderer->setResourceManager(rsrcMgr);
  }
#endif  // MAPDGL

  return renderer;
}

RendererShPtr WindowManager::createRendererForWindow(const RendererSettings& settings, const WindowShPtr& windowPtr) {
  CHECK(_implPtr != nullptr);

  RendererShPtr rendererPtr = createRenderer(settings);
  windowPtr->attachRenderer(rendererPtr.get());

  return rendererPtr;
}

WindowShPtr WindowManager::createWindowWithRenderer(const WindowSettings& settings, const RendererShPtr& rendererPtr) {
  CHECK(_implPtr != nullptr);
  WindowShPtr windowPtr = createWindow(settings);
  windowPtr->attachRenderer(rendererPtr.get());
  return windowPtr;
}

std::pair<WindowShPtr, RendererShPtr> WindowManager::createWindowAndRenderer(const WindowSettings& settings) {
  CHECK(_implPtr != nullptr);
  RendererSettings rendererSettings(settings);

  WindowShPtr windowPtr = createWindow(settings);
  RendererShPtr rendererPtr = createRenderer(rendererSettings);
  windowPtr->attachRenderer(rendererPtr.get());

  return std::make_pair(windowPtr, rendererPtr);
}

void WindowManager::addRendererToWindow(const RendererShPtr& rendererPtr, const WindowShPtr& windowPtr) {
  CHECK(_implPtr != nullptr);
  windowPtr->attachRenderer(rendererPtr.get());
}
void WindowManager::addWindowToRenderer(const WindowShPtr& windowPtr, const RendererShPtr& rendererPtr) {
  CHECK(_implPtr != nullptr);
  windowPtr->attachRenderer(rendererPtr.get());
}

}  // namespace Rendering
