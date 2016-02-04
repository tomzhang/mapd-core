#ifndef RENDERING_GL_EGL_TYPES_H_
#define RENDERING_GL_EGL_TYPES_H_

#include <EGL/egl.h>
#include <memory>

namespace Rendering {
namespace GL {
namespace EGL {

class EglDisplay;
typedef std::weak_ptr<EglDisplay> EglDisplayWkPtr;
typedef std::shared_ptr<EglDisplay> EglDisplayShPtr;

class EglDisplayManager;
class EglWindowManagerImpl;

class EglGLWindow;
typedef std::weak_ptr<EglGLWindow> EglGLWindowWkPtr;
typedef std::shared_ptr<EglGLWindow> EglGLWindowShPtr;

class EglGLRenderer;
typedef std::weak_ptr<EglGLRenderer> EglGLRendererWkPtr;
typedef std::shared_ptr<EglGLRenderer> EglGLRendererShPtr;

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_EGL_TYPES_H_
