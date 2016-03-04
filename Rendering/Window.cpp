#include "Window.h"
#include "Renderer.h"

namespace Rendering {

using Objects::Viewport;
using Settings::WindowSettings;
using Settings::IntSetting;
using Settings::IntConstant;
using Settings::StrSetting;

Window::Window(RendererType type, const WindowSettings& windowSettings, const WindowShPtr& prnt)
    : _initialized(false), _type(type), _settings(windowSettings), _prnt(prnt) {
}

// Window::Window(const WindowSettings& windowSettings, const RendererShPtr& rendererPtr, const WindowShPtr& prnt)
//     : _settings(windowSettings), _prnt(prnt), _rendererPtr(rendererPtr) {
// }

// Window::Window(const WindowShPtr& prnt) : _initialized(false), _settings(), _prnt(prnt), _rendererPtr(nullptr) {
// }

Window::~Window() {
  for (const auto& renderer : _attachedRenderers) {
    renderer->detachWindow(this);
  }
}

void Window::attachRenderer(Renderer* renderer) {
  // TODO(croot): note - only allowing one attached
  // renderer per window right now until I figure
  // out how multi-renderers per window works. I have
  // yet to get that working.
  RUNTIME_EX_ASSERT(
      _attachedRenderers.size() == 0,
      "Cannot attach renderer to window. It already has an attached renderer and only 1 is allowed currently.");

  // TODO(croot): Do we need to make this thread safe?
  for (const auto& item : _attachedRenderers) {
    if (item == renderer) {
      // renderer is already attached
      return;
    }
  }

  // Attach this window to the renderer's list
  renderer->attachWindow(this);

  if (!_attachedRenderers.size() && !isInitialized()) {
    _initialize(renderer);
  }

  if (!renderer->isInitialized()) {
    renderer->initialize();
  }

  _attachedRenderers.push_back(renderer);
}

void Window::detachRenderer(Renderer* renderer, bool calledByRenderer) {
  // TODO(croot): Do we need to make this thread safe?

  if (!calledByRenderer) {
    renderer->detachWindow(this);
  }

  RendererArray::iterator it;
  for (it = _attachedRenderers.begin(); it != _attachedRenderers.end(); ++it) {
    if (*it == renderer) {
      _attachedRenderers.erase(it);
      return;
    }
  }
}

Renderer* Window::getPrimaryRenderer() {
  if (_attachedRenderers.size()) {
    return _attachedRenderers[0];
  }

  return nullptr;
}

Window::RendererArray Window::getAllRenderers() {
  return _attachedRenderers;
}

int Window::numRenderers() const {
  return _attachedRenderers.size();
}

void Window::_initialize(Renderer* renderer) {
  RUNTIME_EX_ASSERT(renderer != nullptr, "Cannot initialize a window with an undefined renderer.");
  _init(renderer);
  _initialized = true;
}

// void Window::setRenderer(const RendererShPtr& rendererPtr) {
//   RUNTIME_EX_ASSERT(_rendererPtr == nullptr, "A renderer has already been defined for the window.");
//   _rendererPtr = rendererPtr;
// }

WindowSettings Window::getWindowSettings() const {
  return _settings;
}

const WindowSettings& Window::getWindowSettingsRef() const {
  return _settings;
}

int64_t Window::getIntSetting(const std::string& setting) const {
  return _settings.getIntSetting(setting);
}

int64_t Window::getIntSetting(IntSetting setting) const {
  return _settings.getIntSetting(setting);
}

IntConstant Window::getIntSettingAsConstant(const std::string& setting) const {
  return _settings.getIntSettingAsConstant(setting);
}

IntConstant Window::getIntSettingAsConstant(IntSetting setting) const {
  return _settings.getIntSettingAsConstant(setting);
}

std::string Window::getStrSetting(const std::string& setting) const {
  return _settings.getStrSetting(setting);
}

std::string Window::getStrSetting(StrSetting setting) const {
  return _settings.getStrSetting(setting);
}

void Window::setViewport(const Viewport& viewport) {
  setViewport(viewport.getXPos(), viewport.getYPos(), viewport.getWidth(), viewport.getHeight());
}

void Window::setViewport(int x, int y, int w, int h) {
  _settings.setViewport(x, y, w, h);
}

Viewport Window::getViewport() const {
  return _settings.getViewport();
}

int Window::getWidth() const {
  return _settings.getViewport().getWidth();
}

int Window::getHeight() const {
  return _settings.getViewport().getHeight();
}

WindowShPtr Window::getParent() const {
  return WindowShPtr(_prnt);
}

}  // namespace Rendering
