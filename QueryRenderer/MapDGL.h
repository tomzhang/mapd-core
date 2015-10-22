#ifndef MAPDGL_H_
#define MAPDGL_H_

#include <GL/glew.h>
#include <stdexcept>

namespace MapD_GL {

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

struct GLError : std::runtime_error {
  GLError(GLenum err, const std::string& errstr) : ::std::runtime_error(errstr), code(err){};
  const GLenum code;
};

void checkGLError(const char* cmd, const char* file, int line);

#if defined(DEBUG)
#define MAPD_CHECK_GL_ERROR(cmd)                                          \
  ([&]() {                                                                \
    struct __MAPD_GL_ERR {                                                \
      inline ~__MAPD_GL_ERR() { checkGLError(#cmd, __FILE__, __LINE__); } \
    } __MAPD_GL_ERR;                                                      \
    return cmd;                                                           \
  }())
#else
#define MAPD_CHECK_GL_ERROR(cmd) (cmd)
#endif

}  // namespace MapD_GL

#endif  // MAPDGL_H_
