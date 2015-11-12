#ifndef MAPDGL_H_
#define MAPDGL_H_

#include <stdexcept>

namespace MapD_GL {

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

void checkGLError(const char* cmd, const char* file, int line);

#ifndef NDEBUG
#define MAPD_CHECK_GL_ERROR(cmd)                                                   \
  ([&]() {                                                                         \
    struct __MAPD_GL_ERR {                                                         \
      inline ~__MAPD_GL_ERR() { MapD_GL::checkGLError(#cmd, __FILE__, __LINE__); } \
    } __MAPD_GL_ERR;                                                               \
    return cmd;                                                                    \
  }())
#else
#define MAPD_CHECK_GL_ERROR(cmd) (cmd)
#endif

}  // namespace MapD_GL

#endif  // MAPDGL_H_
