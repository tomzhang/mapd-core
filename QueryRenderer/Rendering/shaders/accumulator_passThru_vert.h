#ifndef ACCUMULATOR_PASSTHRU_VERT_H_
#define ACCUMULATOR_PASSTHRU_VERT_H_

#include <string>

namespace QueryRenderer {
struct Accumulator_PassThru_vert {
  static const std::string source;
};

const std::string Accumulator_PassThru_vert::source =
    "// VERTEX SHADER\n"
    "#version 450 core\n"
    "\n"
    "in vec2 pos;\n"
    "void main() {\n"
    "    gl_Position = vec4(pos.x, pos.y, 0.5, 1.0);\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // ACCUMULATOR_PASSTHRU_VERT_H_
