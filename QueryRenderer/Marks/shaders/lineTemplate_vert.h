#ifndef LINETEMPLATE_VERT_H_
#define LINETEMPLATE_VERT_H_

#include <string>

namespace QueryRenderer {
struct LineTemplate_Vert {
  static const std::string source;
};

const std::string LineTemplate_Vert::source =
    "// VERTEX SHADER\n"
    "\n"
    "#version 450 core\n"
    "\n"
    "#define inTx <inTxType>\n"
    "#define outTx <outTxType>\n"
    "\n"
    "#define inTy <inTyType>\n"
    "#define outTy <outTyType>\n"
    "\n"
    "#define usePerVertId <usePerVertId>\n"
    "#if usePerVertId == 1\n"
    "in uint id;\n"
    "#endif\n"
    "\n"
    "\n"
    "in inTx x;\n"
    "in inTy y;\n"
    "\n"
    "#define usePerVertColor <usePerVertColor>\n"
    "#define inTstrokeColor <inTstrokeColorType>\n"
    "#define inTstrokeColorEnum <inTstrokeColorEnum>\n"
    "#define outTstrokeColor <outTstrokeColorType>\n"
    "#define outTstrokeColorEnum <outTstrokeColorEnum>\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "in inTstrokeColor strokeColor;\n"
    "#endif\n"
    "\n"
    "#define usePerVertWidth <usePerVertWidth>\n"
    "#define inTstrokeWidth <inTstrokeWidthType>\n"
    "#define outTstrokeWidth <outTstrokeWidthType>\n"
    "\n"
    "#if usePerVertWidth == 1\n"
    "in inTstrokeWidth strokeWidth;\n"
    "#endif\n"
    "\n"
    "\n"
    "#define useUniformBuffer <useUniformBuffer>\n"
    "#if useUniformBuffer == 1\n"
    "<lineData>\n"
    "#endif\n"
    "\n"
    "outTx getx(in inTx x) {\n"
    "  return x;\n"
    "}\n"
    "\n"
    "outTy gety(in inTy y) {\n"
    "  return y;\n"
    "}\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "outTstrokeColor getstrokeColor(in inTstrokeColor strokeColor) {\n"
    // TODO(croot): create a set of color utility functions that
    // can be added #include -- need to create the utility functions
    // and create an ability to #include utilities
    "#if (inTstrokeColorEnum == INT || inTstrokeColorEnum == UNSIGNED_INT) && outTstrokeColorEnum == FLOAT_VEC4\n"
    "  vec4 color;\n"
    "  color.r = float((strokeColor >> 24) & 0xFF) / 255.0;\n"
    "  color.g = float((strokeColor >> 16) & 0xFF) / 255.0;\n"
    "  color.b = float((strokeColor >> 8) & 0xFF) / 255.0;\n"
    "  color.a = float(strokeColor & 0xFF) / 255.0;\n"
    "  return color;\n"
    "#else\n"
    "  return strokeColor;\n"
    "#endif\n"
    "}\n"
    "#endif\n"
    "\n"
    "#if usePerVertWidth == 1\n"
    "outTstrokeWidth getstrokeWidth(in inTstrokeWidth strokeWidth)  {\n"
    "  return strokeWidth;\n"
    "}\n"
    "#endif\n"
    "\n"
    "// viewport data\n"
    "struct Viewport\n"
    "{\n"
    "    int x;\n"
    "    int y;\n"
    "    int width;\n"
    "    int height;\n"
    "};\n"
    "uniform Viewport viewport;\n"
    "\n"
    "vec2 NDCtoScreen(in outTx x, in outTy y) {\n"
    "  return vec2(float(x + 1) * (float(viewport.width) / 2.0) + viewport.x, float(y + 1) * (float(viewport.height) / "
    "2.0) + viewport.y);\n"
    "}\n"
    "\n"
    "#if usePerVertId == 1\n"
    "out uint gId;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "out vec4 gColor;\n"
    "#endif\n"
    "\n"
    "#if usePerVertWidth == 1\n"
    "out float gWidth;\n"
    "#endif\n"
    "\n"
    "void main() {\n"
    "  gl_Position = vec4(NDCtoScreen(getx(x), gety(y)), 0.5, 1.0);\n"
    "\n"
    "#if usePerVertId\n"
    "  gId = id;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor\n"
    "  gColor = getstrokeColor(strokeColor);\n"
    "#endif\n"
    "\n"
    "#if usePerVertWidth\n"
    "  gWidth = getstrokeWidth(strokeWidth);\n"
    "#endif\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // LINETEMPLATE_VERT_H_
