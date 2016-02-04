#ifndef RENDERING_GL_EGL_EGLUTILS_H_
#define RENDERING_GL_EGL_EGLUTILS_H_

#include "Types.h"
#include "../../../Settings/Types.h"
#include <vector>

namespace Rendering {
namespace GL {
namespace EGL {

void appendFbAttrs(EglGLRenderer* renderer,
                   Settings::IntConstant drawableType,
                   const Settings::BaseSettings& settings,
                   std::vector<int>& attributes);

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_EGL_EGLUTILS_H_
