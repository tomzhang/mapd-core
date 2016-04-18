#ifndef QUERYRENDERER_GLX_COMPOSITORFRAG_H_
#define QUERYRENDERER_GLX_COMPOSITORFRAG_H_

#include <string>

namespace QueryRenderer {
namespace Impl {
namespace GLX {

struct Compositor_frag {
  static const std::string source;
};

const std::string Compositor_frag::source =
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(location = 1) out uint id;\n"
    "\n"
    "uniform sampler2DArray rgbaArraySampler;\n"
    "uniform int rgbaArraySize;\n"
    "\n"
    "uniform usampler2DArray idArraySampler;\n"
    "uniform int idArraySize;\n"
    "\n"
    "in vec2 oTexCoords;\n"
    "\n"
    "void main() {\n"
    "    vec4 dstColor = vec4(0,0,0,0);\n"
    "    vec4 srcColor = vec4(0,0,0,0);\n"
    "    float srcFactorRGB;\n"
    "    float dstFactorRGB;\n"
    "    float srcFactorA = 1.0;\n"
    "    float dstFactorA;\n"
    "\n"
    "    uint primitveId = 0;\n"
    "    int i = 0;\n"
    "\n"
    "    while (i < rgbaArraySize) {\n"
    "\n"
    "      // doing our own blend here to avoid mulitple texture lookups.\n"
    "      // We're doing the same blend according to:\n"
    "      // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)\n"
    "      // glBlendEquationsSeparate(GL_ADD, GL_ADD);\n"
    "      // This is what is beind done in the query rendering, but if that\n"
    "      // were to ever change, we'd have to change that here as well.\n"
    "      // We should look at ways of exposing all those options as subroutines.\n"
    "\n"
    "      srcColor = texture(rgbaArraySampler, vec3(oTexCoords, i));\n"
    "\n"
    "      if (srcColor.a > 0.0) {\n"
    "        if (srcColor.a < 1.0) {\n"
    "          if (dstColor.a > 0.0) {\n"
    "            srcFactorRGB = srcColor.a;\n"
    "            dstFactorRGB = 1 - srcColor.a;\n"
    "            dstFactorA = dstFactorRGB;\n"
    "\n"
    "            dstColor = vec4(srcFactorRGB * srcColor.rgb + dstFactorRGB * dstColor.rgb, srcFactorA * srcColor.a + "
    "dstFactorA * dstColor.a);\n"
    "\n"
    "            if (dstColor.a >= 1.0) {\n"
    "              break;\n"
    "            }\n"
    "          } else {\n"
    "            dstColor = srcColor;\n"
    "          }\n"
    "        } else {\n"
    "          dstColor = srcColor;\n"
    "          break;\n"
    "        }\n"
    "      }\n"
    "\n"
    "      ++i;\n"
    "    }\n"
    "\n"
    "    if (idArraySize > 0) {\n"
    "      primitveId = texture(idArraySampler, vec3(oTexCoords, i)).r;\n"
    "    }\n"
    "\n"
    "    color = dstColor;\n"
    "    id = primitveId;\n"
    "}\n";

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_GLX_COMPOSITORFRAG_H_