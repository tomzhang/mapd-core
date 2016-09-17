#include "Types.h"

namespace Rendering {

namespace GL {

std::string to_string(const UniqueResourceId& id) {
  return "<Renderer id: " + std::to_string(id.first) + ", Resource id: " + std::to_string(id.second) + ">";
}

}  // namespace GL

}  // namespace Rendering
