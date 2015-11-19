#ifndef MAPDGL_H_
#define MAPDGL_H_

#include <stdexcept>

namespace MapD_GL {

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

void checkGLError(const char* cmd, const char* file, int line);

#ifndef NDEBUG
#define MAPD_CHECK_GL_ERROR(cmd)                                          \
  ([&]() {                                                                \
    struct MAPD_GL_ERR {                                                  \
      ~MAPD_GL_ERR() { MapD_GL::checkGLError(#cmd, __FILE__, __LINE__); } \
    } MAPD_GL_ERR_INST;                                                   \
    (void) MAPD_GL_ERR_INST;                                              \
    return cmd;                                                           \
  }())
#else
#define MAPD_CHECK_GL_ERROR(cmd) (cmd)
#endif

}  // namespace MapD_GL

#endif  // MAPDGL_H_
