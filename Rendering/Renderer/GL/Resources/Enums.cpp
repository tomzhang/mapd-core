#include "Enums.h"
#include "../../../RenderError.h"

namespace Rendering {
namespace GL {
namespace Resources {

std::string to_string(const GLResourceType value) {
  switch (value) {
    case GLResourceType::SHADER:
      return "SHADER";
    case GLResourceType::FRAMEBUFFER:
      return "FRAMEBUFFER";
    case GLResourceType::RENDERBUFFER:
      return "RENDERBUFFER";
    case GLResourceType::TEXTURE_2D:
      return "TEXTURE_2D";
    case GLResourceType::TEXTURE_2D_ARRAY:
      return "TEXTURE_2D_ARRAY";
    case GLResourceType::VERTEX_BUFFER:
      return "VERTEX_BUFFER";
    case GLResourceType::INDEX_BUFFER:
      return "INDEX_BUFFER";
    case GLResourceType::UNIFORM_BUFFER:
      return "UNIFORM_BUFFER";
    case GLResourceType::INDIRECT_DRAW_VERTEX_BUFFER:
      return "INDIRECT_DRAW_VERTEX_BUFFER";
    case GLResourceType::INDIRECT_DRAW_INDEX_BUFFER:
      return "INDIRECT_DRAW_INDEX_BUFFER";
    case GLResourceType::PIXEL_BUFFER_2D:
      return "PIXEL_BUFFER_2D";
    case GLResourceType::VERTEXARRAY:
      return "VERTEXARRAY";
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

std::string to_string(const BufferAccessType value) {
  switch (value) {
    case BufferAccessType::READ:
      return "READ";
    case BufferAccessType::WRITE:
      return "WRITE";
    case BufferAccessType::READ_AND_WRITE:
      return "READ_AND_WRITE";
    case BufferAccessType::COPY:
      return "COPY";
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

std::string to_string(const BufferAccessFreq value) {
  switch (value) {
    case BufferAccessFreq::STREAM:
      return "STREAM";
    case BufferAccessFreq::STATIC:
      return "STATIC";
    case BufferAccessFreq::DYNAMIC:
      return "DYNAMIC";
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

GLenum getBufferUsage(BufferAccessType accessType, BufferAccessFreq accessFreq) {
  switch (accessFreq) {
    case BufferAccessFreq::STREAM:
      switch (accessType) {
        case BufferAccessType::READ:
          return GL_STREAM_READ;
        case BufferAccessType::WRITE:
          return GL_STREAM_DRAW;
        case BufferAccessType::READ_AND_WRITE:
          return GL_STREAM_DRAW;
        case BufferAccessType::COPY:
          return GL_STREAM_COPY;
        default:
          THROW_RUNTIME_EX("Unsupported buffer access type: " + to_string(accessType));
      }
    case BufferAccessFreq::STATIC:
      switch (accessType) {
        case BufferAccessType::READ:
          return GL_STATIC_READ;
        case BufferAccessType::WRITE:
          return GL_STATIC_DRAW;
        case BufferAccessType::READ_AND_WRITE:
          return GL_STATIC_DRAW;
        case BufferAccessType::COPY:
          return GL_STATIC_COPY;
        default:
          THROW_RUNTIME_EX("Unsupported buffer access type: " + to_string(accessType));
      }

    case BufferAccessFreq::DYNAMIC:
      switch (accessType) {
        case BufferAccessType::READ:
          return GL_DYNAMIC_READ;
        case BufferAccessType::WRITE:
          return GL_DYNAMIC_DRAW;
        case BufferAccessType::READ_AND_WRITE:
          return GL_DYNAMIC_DRAW;
        case BufferAccessType::COPY:
          return GL_DYNAMIC_COPY;
        default:
          THROW_RUNTIME_EX("Unsupported buffer access type: " + to_string(accessType));
      }

    default:
      THROW_RUNTIME_EX("Unsupported buffer access frequency type: " + to_string(accessFreq));
  }

  return 0;
}

}  // namespace Rendering

}  // namespace Rendering

}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLResourceType value) {
  os << Rendering::GL::Resources::to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::BufferAccessType value) {
  os << Rendering::GL::Resources::to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::BufferAccessFreq value) {
  os << Rendering::GL::Resources::to_string(value);
  return os;
}
