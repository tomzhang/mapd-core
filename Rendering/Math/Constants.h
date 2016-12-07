#ifndef RENDERING_MATH_CONSTANTS_H
#define RENDERING_MATH_CONSTANTS_H

#include <algorithm>

namespace Rendering {
namespace Math {

const double PI_d = 3.141592653589793;
const float PI_f = 3.1415927;

template <typename T>
inline T get_pi() {}

template <>
inline double get_pi() {
  return PI_d;
}

template <>
inline float get_pi() {
  return PI_f;
}

const double RADTODEG_d = 180.0 / PI_d;
const float RADTODEG_f = 180.0f / PI_f;

const double DEGTORAD_d = PI_d / 180.0;
const float DEGTORAD_f = PI_f / 180.0f;

inline double rad2deg(const double rad) {
  return rad * RADTODEG_d;
}

inline float rad2deg(const float rad) {
  return rad * RADTODEG_f;
}

inline double deg2rad(const double deg) {
  return deg * DEGTORAD_d;
}

inline float deg2rad(const float deg) {
  return deg * DEGTORAD_f;
}

template <typename T>
T clamp(const T& val, const T& lo, const T& hi) {
  return std::max(lo, std::min(val, hi));
}

}  // namespace Math

}  // namespace Rendering

#endif  // RENDERING_MATH_CONSTANTS_H
