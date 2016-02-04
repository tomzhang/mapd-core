#ifndef RENDERING_GL_GLX_GLXUTILS_H_
#define RENDERING_GL_GLX_GLXUTILS_H_

#include "Types.h"
#include "../../../Settings/Types.h"
#include <vector>

namespace Rendering {
namespace GL {
namespace GLX {

void appendFbBufferAttrs(GlxGLRenderer* renderer,
                         Settings::IntConstant drawableType,
                         const Settings::BaseSettings& settings,
                         std::vector<int>& attributes);

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLX_GLXUTILS_H_
