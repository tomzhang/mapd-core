#include "Types.h"

namespace QueryRenderer {

std::string to_string(const ScaleType scaleType) {
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

std::string to_string(const AccumulatorType accumType) {
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

std::string to_string(const ScaleInterpType interpType) {
  switch (interpType) {
    case ScaleInterpType::InterpolateRgb:
      return "InterpolateRGB";
    case ScaleInterpType::InterpolateHsl:
      return "InterpolateHsl";
    case ScaleInterpType::InterpolateHslLong:
      return "InterpolateHslLong";
    case ScaleInterpType::InterpolateLab:
      return "InterpolateLab";
    case ScaleInterpType::InterpolateHcl:
      return "InterpolateHcl";
    case ScaleInterpType::InterpolateHclLong:
      return "InterpolateHclLong";
    case ScaleInterpType::UNDEFINED:
      return "UNDEFINED";
    default:
      return "scale interpolator type " + std::to_string(static_cast<int>(interpType)) + ">";
  }
  return "";
}

std::vector<std::string> getScaleInterpTypes(const std::vector<ScaleInterpType>& interps) {
  int numInterpTypes = interps.size();
  bool useArg = numInterpTypes > 0;
  if (!useArg) {
    numInterpTypes = static_cast<int>(ScaleInterpType::UNDEFINED) - 1;
  }

  std::vector<std::string> rtn(numInterpTypes);
  if (useArg) {
    int i = 0;
    for (auto& interp : interps) {
      rtn[i++] = to_string(interp);
    }
  } else {
    for (int i = 0; i < numInterpTypes; ++i) {
      rtn[i] = to_string(static_cast<ScaleInterpType>(i));
    }
  }
  return rtn;
}

bool isQuantitativeScale(const ScaleType type) {
  return (type == ScaleType::LINEAR || type == ScaleType::LOG || type == ScaleType::POW || type == ScaleType::SQRT);
}

}  // namespace QueryRenderer
