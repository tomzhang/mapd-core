#ifndef QUERYRENDERER_MULTIGPUCOMPOSITEVERT_H_
#define QUERYRENDERER_MULTIGPUCOMPOSITEVERT_H_

#include <string>

namespace QueryRenderer {

struct MultiGpuComposite_vert {
  static const std::string source;
};

const std::string MultiGpuComposite_vert::source =
    "// VERTEX SHADER\n"
    "\n"
    "#version 450 core\n"
    "\n"
    "in vec2 pos;\n"
    "\n"
    "void main() {\n"
    "    gl_Position = vec4(pos.x, pos.y, 0.5, 1.0);\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MULTIGPUCOMPOSITEVERT_H_
