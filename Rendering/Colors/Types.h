#ifndef RENDERING_COLORS_TYPES_H_
#define RENDERING_COLORS_TYPES_H_

#include "../Math/Constants.h"

#include <cmath>
#include <array>
#include <type_traits>

namespace Rendering {
namespace Colors {

namespace Validators {

template <typename T>
struct PassThruValidation {
  inline T operator()(const T& v) const { return v; }
};

template <typename T, const int defaultLo, const int defaultHi>
struct ClampValidation {
  ClampValidation() : lo(static_cast<T>(defaultLo)), hi(static_cast<T>(defaultHi)) {}
  ClampValidation(const T& lo, const T& hi) : lo(lo), hi(hi) {}
  T operator()(const T& v) const { return Math::clamp<T>(v, lo, hi); }

 private:
  T lo;
  T hi;
};

template <typename T, const int defaultLo, const int defaultHi>
struct WrapValidation {
  WrapValidation() : lo(static_cast<T>(defaultLo)), hi(static_cast<T>(defaultHi)) {}
  WrapValidation(const T& lo, const T& hi) : lo(lo), hi(hi) {}
  T operator()(const T& v) const {
    auto val = v;
    if (val < lo) {
      val = hi - std::fmod(lo - val, hi - lo);
    }
    return lo + std::fmod(val - lo, hi - lo);
  }

 private:
  T lo;
  T hi;
};

typedef ClampValidation<float, 0, 1> Clamp0to1f;
typedef WrapValidation<float, 0, 360> AngleWrap0to360;

}  // namespace Validators

namespace PackedFloatConverters {
struct ConvertUInt8To0to1 {
  float operator()(const uint8_t& val) const { return Math::clamp<float>(float(val) / 255.0f, 0.0f, 1.0f); }
  uint8_t inverse(const float val) const { return static_cast<uint8_t>(Math::clamp(val, 0.0f, 1.0f) * 255.0f); }
};
}  // PackedFloatConverters

struct ColorInitializer;

enum class ColorType { RGBA = 0, HSL, LAB, HCL };
typedef std::array<float, 4> ColorArray;
class ColorRGBA;
class ColorHSL;
class ColorLAB;
class ColorHCL;
class ColorUnion;

template <class T>
struct is_color : std::integral_constant<bool,
                                         std::is_same<ColorRGBA, typename std::remove_cv<T>::type>::value ||
                                             std::is_same<ColorHSL, typename std::remove_cv<T>::type>::value ||
                                             std::is_same<ColorLAB, typename std::remove_cv<T>::type>::value ||
                                             std::is_same<ColorHCL, typename std::remove_cv<T>::type>::value> {};

template <class T, class TT>
struct is_specific_color : std::integral_constant<bool, std::is_same<TT, typename std::remove_cv<T>::type>::value> {};

template <class T>
struct is_color_union
    : std::integral_constant<bool, std::is_same<ColorUnion, typename std::remove_cv<T>::type>::value> {};

template <class T, typename std::enable_if<is_color<T>::value>::type* = nullptr>
ColorType getColorType() {
  return ColorType::RGBA;
}

template <>
ColorType getColorType<ColorRGBA>();

template <>
ColorType getColorType<ColorHSL>();

template <>
ColorType getColorType<ColorLAB>();

template <>
ColorType getColorType<ColorHCL>();

std::string to_string(const ColorType type);

}  // namespace Colors
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::Colors::ColorType colorType);

#endif  // RENDERING_COLORS_TYPES_H_
