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

std::string getGLErrorStr() {
  const GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    switch (err) {
      case GL_INVALID_ENUM:
        return "INVALID_ENUM";
      case GL_INVALID_VALUE:
        return "INVALID_VALUE";
      case GL_INVALID_OPERATION:
        return "INVALID_OPERATION";
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "INVALID_FRAMEBUFFER_OPERATION";
      case GL_OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";
      case GL_STACK_UNDERFLOW:
        return "STACK_UNDERFLOW";
      case GL_STACK_OVERFLOW:
        return "STACK_OVERFLOW";
      default:
        return "(unknown)";
    }
  }

  return "";
}

void checkGLError(const char* cmd, const char* file, int line) {
  std::string errStr = getGLErrorStr();
  if (errStr.length()) {
    THROW_RUNTIME_EX(std::string(cmd) + ": Error GL_" + errStr + " - " + file + ":" + std::to_string(line));
  }
}

}  // namespace GL

}  // namespace Rendering
