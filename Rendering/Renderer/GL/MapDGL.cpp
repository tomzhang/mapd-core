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

std::string getGLErrorStr(const GLenum err) {
  if (err != GL_NO_ERROR) {
    switch (err) {
      case GL_INVALID_ENUM:
        return "GL_INVALID_ENUM";
      case GL_INVALID_VALUE:
        return "GL_INVALID_VALUE";
      case GL_INVALID_OPERATION:
        return "GL_INVALID_OPERATION";
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "GL_INVALID_FRAMEBUFFER_OPERATION";
      case GL_OUT_OF_MEMORY:
        return "GL_OUT_OF_MEMORY";
      case GL_STACK_UNDERFLOW:
        return "GL_STACK_UNDERFLOW";
      case GL_STACK_OVERFLOW:
        return "GL_STACK_OVERFLOW";
      default: {
        std::stringstream ss;
        ss << std::hex << err;
        return "(unknown gl error: " + ss.str() + ")";
      }
    }
  }

  return "";
}

std::string getGLErrorStr() {
  return getGLErrorStr(glGetError());
}

void checkGLError(const char* cmd, const char* file, int line) {
  const GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    std::string errStr = getGLErrorStr(err);
    if (err != GL_OUT_OF_MEMORY) {
      THROW_RUNTIME_EX(std::string(cmd) + ": Error " + errStr + " - " + file + ":" + std::to_string(line));
    } else {
      LOG(ERROR) << errStr << " OpenGL error - " << cmd << ": " << file << ":" << line;
      throw ::Rendering::OutOfGpuMemoryError(errStr + " OpenGL error");
    }
  }
}

void checkGLMemError(const char* file, int line) {
  GLenum err = GL_NO_ERROR;

  bool errorsLogged = false;
  while ((err = glGetError()) != GL_NO_ERROR) {
    if (err != GL_OUT_OF_MEMORY && !errorsLogged) {
      LOG(WARNING) << "More than 1 OpenGL error found in error stack when checking for GL_OUT_OF_MEMORY errors. It is "
                      "recommended to run with a debug build to activate full opengl error checking. The error at the "
                      "top the stack is: " << getGLErrorStr(err);
      errorsLogged = true;
    }
  }

  if (err == GL_OUT_OF_MEMORY) {
    LOG(ERROR) << "GL_OUT_OF_MEMORY OpenGL error - " << file << ":" << line;
    throw ::Rendering::OutOfGpuMemoryError("GL_OUT_OF_MEMORY OpenGL error");
  }
}

}  // namespace GL

}  // namespace Rendering
