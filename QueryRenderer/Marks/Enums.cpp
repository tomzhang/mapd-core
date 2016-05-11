#include "Enums.h"
#include "../Utils/Utils.h"

namespace QueryRenderer {

std::string to_string(const LineJoinType value) {
  switch (value) {
    case LineJoinType::BEVEL:
      return "BEVEL";
    case LineJoinType::ROUND:
      return "ROUND";
    case LineJoinType::MITER:
      return "MITER";
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

bool isLineJoinEnum(const int val) {
  return (val >= static_cast<int>(LineJoinType::BEVEL) && val <= static_cast<int>(LineJoinType::MITER));
}

bool isLineJoinEnum(const std::string& val) {
  return false;
}

int convertStringToLineJoinEnum(const std::string& val) {
  std::string uppercase = makeUpperCase(val);

  if (uppercase == to_string(LineJoinType::BEVEL)) {
    return static_cast<int>(LineJoinType::BEVEL);
  } else if (uppercase == to_string(LineJoinType::ROUND)) {
    return static_cast<int>(LineJoinType::ROUND);
  } else if (uppercase == to_string(LineJoinType::MITER)) {
    return static_cast<int>(LineJoinType::MITER);
  }

  return -1;
}

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::LineJoinType value) {
  os << QueryRenderer::to_string(value);
  return os;
}
