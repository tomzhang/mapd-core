#ifndef COLOR_H_
#define COLOR_H_

#include <array>
#include <string>

namespace MapD_Renderer {

enum ColorChannel { RED=0, GREEN, BLUE, ALPHA };

class ColorRGBA {
    public:
        ColorRGBA() : _colorArray({0,0,0,255}) {}

        ColorRGBA(float r, float g, float b);
        ColorRGBA(float r, float g, float b, float a);

        ColorRGBA(int r, int g, int b);
        ColorRGBA(int r, int g, int b, int a);

        ColorRGBA(char r, char g, char b) : _colorArray({r, g, b, 255}) {}
        ColorRGBA(char r, char g, char b, char a) : _colorArray({r, g, b, a}) {}

        ColorRGBA(const ColorRGBA& color) : _colorArray(color._colorArray) {}

        ColorRGBA& operator=(const ColorRGBA& rhs) {
            _colorArray = rhs._colorArray;
            return *this;
        }

        operator std::string() const {
            std::string = "[" + ;
            for (auto channel : _colorArray) {

            }
            s += "]";
            return s;
        }
    private:
        std::array<char, 4> _colorArray;
};

} // namespace MapD_Renderer

#endif // COLOR_H_
