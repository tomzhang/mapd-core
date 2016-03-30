#include "Types.h"

namespace QueryRenderer {

std::string to_string(const UserWidgetIdPair& value) {
  return "[userId: " + std::to_string(value.userId) + ", widgetId: " + std::to_string(value.widgetId) + "]";
}

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::UserWidgetIdPair& value) {
  os << QueryRenderer::to_string(value);
  return os;
}
