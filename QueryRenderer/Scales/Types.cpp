#include "Types.h"

namespace QueryRenderer {

std::string to_string(ScaleType scaleType) {
  switch (scaleType) {
    case ScaleType::LINEAR:
      return "LINEAR";
    case ScaleType::ORDINAL:
      return "ORDINAL";
    case ScaleType::QUANTIZE:
      return "QUANTIZE";
    case ScaleType::UNDEFINED:
      return "UNDEFINED";
    default:
      return "<scale type " + std::to_string(static_cast<int>(scaleType)) + ">";
  }

  return "";
}

}  // namespace QueryRenderer
