#ifndef QUERYRENDERER_GLX_COMPOSITORVERT_H_
#define QUERYRENDERER_GLX_COMPOSITORVERT_H_

#include <string>

namespace QueryRenderer {
namespace Impl {
namespace GLX {

struct Compositor_vert {
  static const std::string source;
};

const std::string Compositor_vert::source =
    "// VERTEX SHADER\n"
    "\n"
    "#version 450 core\n"
    "\n"
    "in vec2 pos;\n"
    "\n"
    "void main() {\n"
    "    gl_Position = vec4(pos.x, pos.y, 0.5, 1.0);\n"
    "}\n";
}  // namespace GLX
}  // namespace Impl

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_GLX_COMPOSITORVERT_H_
