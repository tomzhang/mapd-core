#include "Types.h"

namespace QueryRenderer {

std::string to_string(ScaleType scaleType) {
  switch (scaleType) {
    case ScaleType::LINEAR:
      return "LINEAR";
    case ScaleType::LOG:
      return "LOG";
    case ScaleType::POW:
      return "POW";
    case ScaleType::SQRT:
      return "SQRT";
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

std::string to_string(AccumulatorType accumType) {
  switch (accumType) {
    case AccumulatorType::MIN:
      return "MIN";
    case AccumulatorType::MAX:
      return "MAX";
    case AccumulatorType::BLEND:
      return "BLEND";
    case AccumulatorType::DENSITY:
      return "DENSITY";
    case AccumulatorType::UNDEFINED:
      return "UNDEFINED";
    case AccumulatorType::ALL:
      return "ALL";
    default:
      return "<scale type " + std::to_string(static_cast<int>(accumType)) + ">";
  }

  return "";
}

bool isQuantitativeScale(const ScaleType type) {
  return (type == ScaleType::LINEAR || type == ScaleType::LOG || type == ScaleType::POW || type == ScaleType::SQRT);
}

}  // namespace QueryRenderer
