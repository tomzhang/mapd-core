#include "Types.h"

namespace QueryRenderer {

std::string to_string(const UserWidgetIdPair& value) {
  return "[userId: " + value.userId + ", widgetId: " + std::to_string(value.widgetId) + "]";
}

std::string to_string(const RefType refType) {
  switch (refType) {
    case RefType::DATA:
      return "DATA";
    case RefType::SCALE:
      return "SCALE";
    default:
      return "UNKNOWN";
  }

  return "";
}

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::UserWidgetIdPair& value) {
  os << QueryRenderer::to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const QueryRenderer::RefType refType) {
  os << QueryRenderer::to_string(refType);
  return os;
}
