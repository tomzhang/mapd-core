#ifndef POINTTEMPLATE_FRAG_H_
#define POINTTEMPLATE_FRAG_H_

#include <string>

namespace MapD_Renderer {
struct PointTemplate_Frag {
  static const std::string source;
};

const std::string PointTemplate_Frag::source =
    "#version 410 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(location = 1) out uint id;\n"
    "\n"
    "flat in uint fPrimitiveId;\n"
    "flat in vec4 fColor;\n"
    "\n"
    "void main() {\n"
    "    float dist = distance(gl_PointCoord, vec2(0.5));\n"
    "    //if (dist > 0.5) {\n"
    "    //    discard;\n"
    "    //}\n"
    "\n"
    "    float delta = fwidth(dist);\n"
    "    float alpha = 1.0 - smoothstep(0.5-delta, 0.5, dist);\n"
    "    if (alpha == 0.0) {\n"
    "        discard;\n"
    "    }\n"
    "\n"
    "    color = vec4(fColor.xyz, fColor.w*alpha);\n"
    "    //color = fColor;\n"
    "    id = fPrimitiveId;\n"
    "}\n";
}  // namespace MapD_Renderer

#endif  // POINTTEMPLATE_FRAG_H_
