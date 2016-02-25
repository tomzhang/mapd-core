#ifndef RENDERING_GL_RESOURCES_ENUMS_H_
#define RENDERING_GL_RESOURCES_ENUMS_H_

#include <GL/glew.h>
#include <string>

namespace Rendering {
namespace GL {
namespace Resources {

enum class GLResourceType {
  SHADER = 0,
  FRAMEBUFFER,
  RENDERBUFFER,
  TEXTURE_2D,
  TEXTURE_2D_ARRAY,
  VERTEXBUFFER,
  VERTEXARRAY
};

enum class GLBufferType { VERTEX_BUFFER = 0 };

enum class FboBind { READ = GL_READ_FRAMEBUFFER, DRAW = GL_DRAW_FRAMEBUFFER, READ_AND_DRAW = GL_FRAMEBUFFER };

std::string to_string(const GLResourceType value);

}  // namespace GL
}  // namespace Resources
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLResourceType value);

#endif  // RENDERING_GL_RESOURCES_ENUMS_H_
