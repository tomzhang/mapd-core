#ifndef RENDERING_TYPES_H
#define RENDERING_TYPES_H

#include <memory>

namespace Rendering {

// class ColorRGBA;

// class Viewport;
// typedef std::shared_ptr<Viewport> ViewportShPtr;
// typedef std::unique_ptr<Viewport> ViewportUqPtr;

enum class RendererClass { GPU = 0, CPU };

enum class RendererType { OPENGL = 0 };

class Renderer;
typedef std::shared_ptr<Renderer> RendererShPtr;
typedef std::unique_ptr<Renderer> RendererUqPtr;
typedef std::weak_ptr<Renderer> RendererWkPtr;

class Window;
typedef std::shared_ptr<Window> WindowShPtr;
typedef std::unique_ptr<Window> WindowUqPtr;

class WindowManager;
class WindowManagerImpl;

}  // namespace Rendering

#endif  // RENDERING_TYPES_H
