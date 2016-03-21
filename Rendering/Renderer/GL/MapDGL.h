#ifndef RENDERING_GL_MAPDGL_H_
#define RENDERING_GL_MAPDGL_H_

#include <GL/glew.h>
#include <stdexcept>

namespace Rendering {

namespace GL {

#ifdef GLEW_MX
// #define glewGetContext
GLEWContext* glewGetContext();
#endif

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

std::string getGLErrorStr();
void checkGLError(const char* cmd, const char* file, int line);

#ifndef NDEBUG
#define MAPD_CHECK_GL_ERROR(cmd)                                                \
  ([&]() {                                                                      \
    struct MAPD_GL_ERR {                                                        \
      ~MAPD_GL_ERR() { Rendering::GL::checkGLError(#cmd, __FILE__, __LINE__); } \
    } MAPD_GL_ERR_INST;                                                         \
    (void) MAPD_GL_ERR_INST;                                                    \
    return cmd;                                                                 \
  }())
#else
#define MAPD_CHECK_GL_ERROR(cmd) (cmd)
#endif

}  // namespace GL

}  // namespace Rendering

#endif  // RENDERING_GL_MAPDGL_H_
