#ifndef RENDERING_COLORS_UTILS_H_
#define RENDERING_COLORS_UTILS_H_

#include "Types.h"
#include "../RenderError.h"

#include <regex>

namespace Rendering {
namespace Colors {

bool isColorString(const std::string& str);
ColorType getColorTypeFromColorString(const std::string& str);
ColorType getColorTypeFromColorPrefix(const std::string& colorPrefix);
std::vector<std::string> getAllColorPrefixes();

template <typename T>
T createColorFromColorString(const std::string& str) {}

template <typename T>
void convertColor(const T& from, T& to) {
  to = from;
}

void convertColor(const ColorRGBA& from, ColorHSL& to);
void convertColor(const ColorRGBA& from, ColorLAB& to);
void convertColor(const ColorRGBA& from, ColorHCL& to);

void convertColor(const ColorHSL& from, ColorRGBA& to);
void convertColor(const ColorHSL& from, ColorLAB& to);
void convertColor(const ColorHSL& from, ColorHCL& to);

void convertColor(const ColorLAB& from, ColorRGBA& to);
void convertColor(const ColorLAB& from, ColorHSL& to);
void convertColor(const ColorLAB& from, ColorHCL& to);

void convertColor(const ColorHCL& from, ColorRGBA& to);
void convertColor(const ColorHCL& from, ColorHSL& to);
void convertColor(const ColorHCL& from, ColorLAB& to);

}  // namespace Objects
}  // namespace Rendering

#endif  // RENDERING_COLORS_UTILS_H_
