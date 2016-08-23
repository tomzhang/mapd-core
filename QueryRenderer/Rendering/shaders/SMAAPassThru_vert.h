#ifndef SMAAPASSTHRU_VERT_H_
#define SMAAPASSTHRU_VERT_H_

#include <string>

namespace QueryRenderer {
struct SMAAPassThru_vert {
  static const std::string source;
};

const std::string SMAAPassThru_vert::source =
    "#version 450 core\n"
    "\n"
    "in vec2 pos;\n"
    "in vec2 texcoord;\n"
    "\n"
    "out vec2 fTexCoord;\n"
    "\n"
    "uniform vec4 SMAA_RT_METRICS;\n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    fTexCoord = texcoord;\n"
    "    gl_Position = vec4(pos.xy * SMAA_RT_METRICS.xy * 2 - 1, 0.5, 1.0);\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // SMAAPASSTHRU_VERT_H_
