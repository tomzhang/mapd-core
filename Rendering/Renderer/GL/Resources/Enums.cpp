#include "Enums.h"

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
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

}  // namespace Rendering

}  // namespace Rendering

}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLResourceType value) {
  os << Rendering::GL::Resources::to_string(value);
  return os;
}
