#ifndef RENDERING_GL_GLX_TYPES_H_
#define RENDERING_GL_GLX_TYPES_H_

#include <GL/glxew.h>
#include <X11/Xlib.h>
#include <memory>

namespace Rendering {
namespace GL {
namespace GLX {

typedef unsigned long X11ID;

class X11DisplayManager;

typedef std::weak_ptr<Display> X11DisplayWkPtr;
typedef std::shared_ptr<Display> X11DisplayShPtr;

typedef std::pair<X11DisplayShPtr, int> DisplayScreenPair;

typedef std::shared_ptr<GLXFBConfig> FbConfigShPtr;
typedef std::unique_ptr<GLXFBConfig, int (*)(void*)> FbConfigUqPtr;

class GlxGLRenderer;
typedef std::shared_ptr<GlxGLRenderer> GlxGLRendererShPtr;
typedef std::unique_ptr<GlxGLRenderer> GlxGLRendererUqPtr;

class GlxGLWindow;
typedef std::shared_ptr<GlxGLWindow> GlxGLWindowShPtr;
typedef std::unique_ptr<GlxGLWindow> GlxGLWindowUqPtr;

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLX_TYPES_H_
