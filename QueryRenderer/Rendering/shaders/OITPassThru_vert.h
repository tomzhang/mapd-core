#ifndef OITPASSTHRU_VERT_H_
#define OITPASSTHRU_VERT_H_

#include <string>

namespace QueryRenderer {
struct OITPassThru_vert {
  static const std::string source;
};

const std::string OITPassThru_vert::source =
    "#version 450 core\n"
    "\n"
    "in vec2 pos;\n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    gl_Position = vec4(pos.x, pos.y, 0.5, 1.0);\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // OITPASSTHRU_VERT_H_
