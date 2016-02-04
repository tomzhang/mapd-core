#include "MapDGL.h"
#include "GLRenderer.h"
#include "../../RenderError.h"
// #include "GLRenderer.h"
// #include <GL/glew.h>
// #include <string>

namespace Rendering {

namespace GL {

#ifdef GLEW_MX
GLEWContext* glewGetContext() {
  const GLRenderer* curr = GLRenderer::getCurrentThreadRenderer();
  if (curr) {
    // TODO(croot): Need to cast away const-ness here, but that's generally
    // a bad thing.
    return const_cast<GLEWContext*>(curr->glewGetContext());
  } else {
    return nullptr;
  }
}
#endif

void checkGLError(const char* cmd, const char* file, int line) {
  const GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    const char* error = nullptr;
    switch (err) {
      case GL_INVALID_ENUM:
        error = "INVALID_ENUM";
        break;
      case GL_INVALID_VALUE:
        error = "INVALID_VALUE";
        break;
      case GL_INVALID_OPERATION:
        error = "INVALID_OPERATION";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        error = "INVALID_FRAMEBUFFER_OPERATION";
        break;
      case GL_OUT_OF_MEMORY:
        error = "OUT_OF_MEMORY";
        break;
      case GL_STACK_UNDERFLOW:
        error = "STACK_UNDERFLOW";
        break;
      case GL_STACK_OVERFLOW:
        error = "STACK_OVERFLOW";
        break;
      default:
        error = "(unknown)";
        break;
    }

    THROW_RUNTIME_EX(std::string(cmd) + ": Error GL_" + error + " - " + file + ":" + std::to_string(line));
  }
}

}  // namespace GL

}  // namespace Rendering
