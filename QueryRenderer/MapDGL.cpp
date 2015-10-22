#include "MapDGL.h"
#include <stdexcept>
#include <string>

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

    // TODO(croot): create own exception class
    throw MapD_GL::GLError(err, std::string(cmd) + ": Error GL_" + error + " - " + file + ":" + std::to_string(line));
    // throw std::runtime_error(std::string(cmd) + ": Error GL_" + error + " _ " + file + ":" + std::to_string(line));
  }
}
