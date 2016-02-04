#include "MapDEGL.h"
#include "../../../RenderError.h"
#include <EGL/egl.h>

namespace Rendering {
namespace GL {
namespace EGL {

void checkEGLError(const char* cmd, const char* file, int line) {
  const EGLint err = eglGetError();
  if (err != EGL_SUCCESS) {
    const char* error = nullptr;
    switch (err) {
      case EGL_NOT_INITIALIZED:
        error = "NOT_INITIALIZED";
        break;
      case EGL_BAD_ACCESS:
        error = "BAD_ACCESS";
        break;
      case EGL_BAD_ALLOC:
        error = "BAD_ALLOC";
        break;
      case EGL_BAD_ATTRIBUTE:
        error = "BAD_ATTRIBUTE";
        break;
      case EGL_BAD_CONTEXT:
        error = "BAD_CONTEXT";
        break;
      case EGL_BAD_CONFIG:
        error = "BAD_CONFIG";
        break;
      case EGL_BAD_CURRENT_SURFACE:
        error = "BAD_CURRENT_SURFACE";
        break;
      case EGL_BAD_DISPLAY:
        error = "BAD_DISPLAY";
        break;
      case EGL_BAD_SURFACE:
        error = "BAD_SURFACE";
        break;
      case EGL_BAD_MATCH:
        error = "BAD_MATCH";
        break;
      case EGL_BAD_PARAMETER:
        error = "BAD_PARAMETER";
        break;
      case EGL_BAD_NATIVE_PIXMAP:
        error = "BAD_NATIVE_PIXMAP";
        break;
      case EGL_BAD_NATIVE_WINDOW:
        error = "BAD_NATIVE_WINDOW";
        break;
      case EGL_CONTEXT_LOST:
        error = "CONTEXT_LOST";
        break;
      default:
        error = "(unknown)";
        break;
    }

    THROW_RUNTIME_EX(std::string(cmd) + ": Error EGL_" + error + " - " + file + ":" + std::to_string(line));
  }
}

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering
