#ifndef OITCOMPOSITE_FRAG_H_
#define OITCOMPOSITE_FRAG_H_

#include <string>

namespace QueryRenderer {
struct OITComposite_frag {
  static const std::string source;
};

const std::string OITComposite_frag::source =
    "// FRAGMENT SHADER\n"
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "\n"
    "uniform sampler2D accumTex;\n"
    "uniform sampler2D revealageTex;\n"
    "\n"
    "void main() {\n"
    "  float revealage = texelFetch(revealageTex, ivec2(gl_FragCoord.xy), 0).r;\n"
    "  if (revealage == 1.0) {\n"
    "    discard;\n"
    "  }\n"
    "  vec4 accum = texelFetch(accumTex, ivec2(gl_FragCoord.xy), 0);\n"
    "  if (isinf(max(max(abs(accum.r), abs(accum.g)), abs(accum.b)))) {\n"
    "    accum.rgb = vec3(accum.a);\n"
    "  }\n"
    "  color.rgb = accum.rgb / max(accum.a, 1e-5);\n"
    "  color.a = 1.0 - revealage;\n"
    "}\n";
}  // namespace QueryRenderer

#endif  // OITCOMPOSITE_FRAG_H_
