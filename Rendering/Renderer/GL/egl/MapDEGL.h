#ifndef RENDERING_GL_EGL_MAPDEGL_H_
#define RENDERING_GL_EGL_MAPDEGL_H_

#include <stdexcept>

namespace Rendering {
namespace GL {
namespace EGL {

void checkEGLError(const char* cmd, const char* file, int line);

#ifndef NDEBUG
#define MAPD_CHECK_EGL_ERROR(cmd)                                                      \
  ([&]() {                                                                             \
    struct MAPD_EGL_ERR {                                                              \
      ~MAPD_EGL_ERR() { Rendering::GL::EGL::checkEGLError(#cmd, __FILE__, __LINE__); } \
    } MAPD_EGL_ERR_INST;                                                               \
    (void) MAPD_EGL_ERR_INST;                                                          \
    return cmd;                                                                        \
  }())
#else
#define MAPD_CHECK_EGL_ERROR(cmd) (cmd)
#endif

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_EGL_MAPDEGL_H_
