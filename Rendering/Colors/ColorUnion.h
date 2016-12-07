#ifndef RENDERING_COLORS_COLORUNION_H_
#define RENDERING_COLORS_COLORUNION_H_

#include "Types.h"
#include "Utils.h"
#include "ColorRGBA.h"
#include "ColorHSL.h"
#include "ColorLAB.h"
#include "ColorHCL.h"
#include "../RenderError.h"

#ifdef MAPDGL
#include "../Renderer/GL/TypeGL.h"
#endif  // MAPDGL

namespace Rendering {
namespace Colors {

class ColorUnion {
 public:
  union UnionColor {
    ColorRGBA rgba;
    ColorHSL hsl;
    ColorLAB lab;
    ColorHCL hcl;
    UnionColor() {}
    ~UnionColor() {}
  };

  // ColorUnion() : _tag(ColorType::RGBA), colorData(), colorData.rgba(0.0f, 0.0f, 0.0f, 1.0f) {}
  ColorUnion() : _tag(ColorType::RGBA) { colorData.rgba.set(0.0f, 0.0f, 0.0f, 1.0f); }

  explicit ColorUnion(const std::string& colorStr);

  explicit ColorUnion(const float chan0,
                      const float chan1,
                      const float chan2,
                      const float opacity = 1.0,
                      const ColorType& type = ColorType::RGBA);

  explicit ColorUnion(const uint8_t c0,
                      const uint8_t c1,
                      const uint8_t c2,
                      const uint8_t opacity = 255,
                      const ColorType type = ColorType::RGBA);

  explicit ColorUnion(const std::array<uint8_t, 4>& color, const ColorType type = ColorType::RGBA);
  explicit ColorUnion(const uint32_t packedColor, const ColorType type = ColorType::RGBA);
  ColorUnion(const ColorUnion& color);

  ~ColorUnion();

  ColorUnion& operator=(const ColorUnion& rhs);

  void set(const uint8_t c0,
           const uint8_t c1,
           const uint8_t c2,
           const uint8_t opacity = 255,
           const ColorType type = ColorType::RGBA);

  void set(const uint32_t packedColor, const ColorType type = ColorType::RGBA);

  void set(const float chan0,
           const float chan1,
           const float chan2,
           const float opacity = 1.0,
           const ColorType type = ColorType::RGBA);

  float operator[](unsigned int channel) const;

  void initFromCSSString(const std::string& colorStr);
  void initFromPackedUInt(const uint32_t packedVal, const ColorType type = ColorType::RGBA);

  float opacity() const;

  operator std::string() const;

  ColorType getType() const { return _tag; }

  template <typename ColorT, typename std::enable_if<is_color<ColorT>::value>::type* = nullptr>
  ColorT get() const {
    ColorT color;
    switch (_tag) {
      case ColorType::RGBA:
        convertColor(colorData.rgba, color);
        break;
      case ColorType::HSL:
        convertColor(colorData.hsl, color);
        break;
      case ColorType::LAB:
        convertColor(colorData.lab, color);
        break;
      case ColorType::HCL:
        convertColor(colorData.hcl, color);
        break;
    }
    return color;
  }

  std::array<float, 4> getColorArray() const;
  const std::array<float, 4>& getColorArrayRef() const;
  uint32_t getPackedColor() const;

  void convertToType(const ColorType colorType);

  static bool isColorString(const std::string& colorStr) { return Colors::isColorString(colorStr); }
  static bool isValidPackedType(const ColorType colorType);
  static std::vector<std::string> getPackedColorPrefixes();

#ifdef MAPDGL
  bool isValidTypeGL(const ::Rendering::GL::TypeGLShPtr& typeGL);

  static bool isPackedTypeGL(const ::Rendering::GL::TypeGLShPtr& typeGL) { return ColorRGBA::isPackedTypeGL(typeGL); }

  static ::Rendering::GL::TypeGL<float, 4> getTypeGL() { return ::Rendering::GL::TypeGL<float, 4>(); }
  static ::Rendering::GL::TypeGLShPtr getTypeGLPtr() {
    return ::Rendering::GL::TypeGLShPtr(new ::Rendering::GL::TypeGL<float, 4>());
  }

  static ::Rendering::GL::TypeGL<uint32_t> getPackedTypeGL() { return ::Rendering::GL::TypeGL<uint32_t>(); }
  static ::Rendering::GL::TypeGLShPtr getPackedTypeGLPtr() {
    return ::Rendering::GL::TypeGLShPtr(new ::Rendering::GL::TypeGL<uint32_t>());
  }
#endif  // MAPDGL

 private:
  ColorType _tag;

  UnionColor colorData;
};

}  // namespace Colors
}  // namespace Rendering

#endif  // RENDERING_COLORS_COLORUNION_H_
