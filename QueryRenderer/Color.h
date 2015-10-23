#ifndef COLOR_H_
#define COLOR_H_

#include <array>
#include <string>
#include <sstream>
#include <cstdint>
#include <unordered_map>
#include <boost/regex.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_floating_point.hpp>

#include "TypeGL.h"

namespace MapD_Renderer {

enum class ColorChannel { RED = 0, GREEN, BLUE, ALPHA };

template <typename T>
inline float convertToColorChannel(const T& val) {
  BOOST_STATIC_ASSERT((!boost::is_floating_point<T>::value));

  return std::max(0.0, std::min(1.0, float(val) / 255.0));
}

class ColorRGBA {
 public:
  typedef std::array<float, 4> ColorArray;

  static bool isValidTypeGL(const TypeGLShPtr& typeGL) {
    return (typeGL->numComponents() == 4 && typeGL->baseGLType() == GL_FLOAT);
  }

  static const std::unordered_map<std::string, std::string> colorKeywords;
  static const boost::regex rgbRegex;
  static const boost::regex rgbaRegex;
  static const boost::regex hexRegex;

  ColorRGBA() : _colorArray({{0, 0, 0, 255}}) {}
  ColorRGBA(const std::string& colorStr) { initFromCSSString(colorStr); }
  ColorRGBA(uint8_t r, uint8_t g, uint8_t b)
      : _colorArray({{convertToColorChannel(r), convertToColorChannel(g), convertToColorChannel(b), 1.0}}) {}
  ColorRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
      : _colorArray({{convertToColorChannel(r),
                      convertToColorChannel(g),
                      convertToColorChannel(b),
                      convertToColorChannel(a)}}) {}
  ColorRGBA(const ColorRGBA& color) : _colorArray(color._colorArray) {}

  ColorRGBA& operator=(const ColorRGBA& rhs) {
    _colorArray = rhs._colorArray;
    return *this;
  }

  ColorRGBA& initFromCSSString(const std::string& colorStr);
  // void initFromCSSString(const std::string& colorStr) {
  //     // TODO: throw exception instead of assert
  //     _colorArray = { {255,255,0,255} };
  // }

  operator std::string() const {
    std::ostringstream s("[");

    for (size_t i = 0; i < _colorArray.size(); ++i) {
      if (i > 0) {
        s << ", ";
      }

      s << int(_colorArray[i] * 255.0);
    }

    s << "]";
    return s.str();
  }

  std::array<float, 4> getColorArray() const { return _colorArray; }

  static bool isColorString(const std::string& colorStr);

 private:
  ColorArray _colorArray;
};

}  // namespace MapD_Renderer

#endif  // COLOR_H_
