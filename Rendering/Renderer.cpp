#include "Renderer.h"
#include "Window.h"

namespace Rendering {

using Settings::RendererSettings;

RendererId Renderer::rendererCnt = 0;

Renderer::Renderer(const RendererSettings& settings) : _attachedWindows(), _initialized(false), _settings(settings) {
  _id = rendererCnt++;
}

Renderer::~Renderer() {
  for (const auto& window : _attachedWindows) {
    window->detachRenderer(this, true);
  }
}

/**
 * Called to attach a window to the renderer. This
 * should only be called by a Window instance.
 */
void Renderer::attachWindow(Window* window) {
  // TODO(croot): note - only allowing one attached
  // window per renderer right now until I figure
  // out how multi-windows per context works. I have
  // yet to get that working.
  RUNTIME_EX_ASSERT(
      _attachedWindows.size() == 0,
      "Cannot attach window to renderer. It already has an attached window and only 1 is allowed currently.");

  // TODO(croot): Do we need to make this thread safe?
  for (const auto& item : _attachedWindows) {
    if (item == window) {
      // window is already attached
      return;
    }
  }

  RUNTIME_EX_ASSERT(_verifyWindowIsAttachable(window), "Cannot attach window to renderer. Window failed verification.");

  _attachedWindows.push_back(window);

  _windowsDirty = true;
}

/**
 * Called to detach a window from the renderer. This function
 * should only be called by a Window instance.
 */
void Renderer::detachWindow(Window* window) {
  // TODO(croot): Do we need to make this thread safe?
  WindowArray::iterator it;
  for (it = _attachedWindows.begin(); it != _attachedWindows.end(); ++it) {
    if (*it == window) {
      _attachedWindows.erase(it);
      return;
    }
  }

  _windowsDirty = true;
}

Window* Renderer::getPrimaryWindow() {
  return const_cast<Window*>(static_cast<const Renderer&>(*this).getPrimaryWindow());
}

const Window* Renderer::getPrimaryWindow() const {
  if (_attachedWindows.size()) {
    return _attachedWindows[0];
  }

  return nullptr;
}

Renderer::WindowArray Renderer::getAllWindows() {
  return _attachedWindows;
}

int Renderer::numWindows() const {
  return _attachedWindows.size();
}

}  // namespace Rendering
