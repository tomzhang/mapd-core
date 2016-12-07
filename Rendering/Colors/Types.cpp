#include "Types.h"

namespace Rendering {
namespace Colors {

template <>
ColorType getColorType<ColorRGBA>() {
  return ColorType::RGBA;
}

template <>
ColorType getColorType<ColorHSL>() {
  return ColorType::HSL;
}

template <>
ColorType getColorType<ColorLAB>() {
  return ColorType::LAB;
}

template <>
ColorType getColorType<ColorHCL>() {
  return ColorType::HCL;
}

std::string to_string(const ColorType type) {
  switch (type) {
    case ColorType::RGBA:
      return "RGBA";
    case ColorType::HSL:
      return "HSL";
    case ColorType::LAB:
      return "LAB";
    case ColorType::HCL:
      return "HCL";
    default:
      return "Unknown Color";
  }

  return "";
}

}  // namespace Colors
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::Colors::ColorType colorType) {
  os << Rendering::Colors::to_string(colorType);
  return os;
}
