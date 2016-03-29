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
  VERTEX_BUFFER,
  INDEX_BUFFER,
  UNIFORM_BUFFER,
  PIXEL_BUFFER_2D,
  INDIRECT_DRAW_VERTEX_BUFFER,
  INDIRECT_DRAW_INDEX_BUFFER,
  VERTEXARRAY
};

enum class GLBufferType {
  VERTEX_BUFFER = 0,
  PIXEL_BUFFER_2D,
  UNIFORM_BUFFER,
  INDEX_BUFFER,
  INDIRECT_DRAW_VERTEX_BUFFER,
  INDIRECT_DRAW_INDEX_BUFFER
};

enum class FboBind { READ = GL_READ_FRAMEBUFFER, DRAW = GL_DRAW_FRAMEBUFFER, READ_AND_DRAW = GL_FRAMEBUFFER };

// TODO(croot): expose enums for buffer access hints
enum class BufferAccessType { READ = 0, WRITE, READ_AND_WRITE, COPY };
enum class BufferAccessFreq { STREAM = 0, STATIC, DYNAMIC };

GLenum getBufferUsage(BufferAccessType accessType, BufferAccessFreq accessFreq);

std::string to_string(const GLResourceType value);
std::string to_string(const BufferAccessType value);
std::string to_string(const BufferAccessFreq value);

}  // namespace GL
}  // namespace Resources
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLResourceType value);
std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::BufferAccessType value);
std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::BufferAccessFreq value);

#endif  // RENDERING_GL_RESOURCES_ENUMS_H_
