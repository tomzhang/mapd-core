#ifndef POLYTEMPLATE_VERT_H_
#define POLYTEMPLATE_VERT_H_

#include <string>

namespace QueryRenderer {
struct PolyTemplate_Vert {
  static const std::string source;
};

const std::string PolyTemplate_Vert::source =
    "// VERTEX SHADER\n"
    "\n"
    "#version 410 core\n"
    "\n"
    "#define inTx <inTxType>\n"
    "#define outTx <outTxType>\n"
    "\n"
    "#define inTy <inTyType>\n"
    "#define outTy <outTyType>\n"
    "\n"
    "#define useUid <useUid>\n"
    "\n"
    "in inTx x;\n"
    "in inTy y;\n"
    "\n"
    "#if useUid == 1\n"
    "uniform uint id;\n"
    "#endif\n"
    "\n"
    "#define usePerVertColor <usePerVertColor>\n"
    "#define useUfillColor <useUfillColor>\n"
    "#define inTfillColor <inTfillColorType>\n"
    "#define outTfillColor <outTfillColorType>\n"
    "#if useUfillColor == 1\n"
    "uniform inTfillColor fillColor;\n"
    "#elif usePerVertColor == 1\n"
    "in inTfillColor fillColor;\n"
    "#endif\n"
    "\n"
    "#define useUniformBuffer <useUniformBuffer>\n"
    "#if useUniformBuffer == 1\n"
    "layout(std140) uniform PolyData {\n"
    "#if !usePerVertColor && !useUfillColor\n"
    "  inTfillColor fillColor;\n"
    "#endif\n"
    "#if !useUid\n"
    "  uint id;\n"
    "#endif\n"
    "} polyData;\n"
    "#endif\n"
    "\n"
    "\n"
    "outTx getx(in inTx x) {\n"
    "  return x;\n"
    "}\n"
    "\n"
    "outTy gety(in inTy y) {\n"
    "  return y;\n"
    "}\n"
    "\n"
    "outTfillColor getfillColor(in inTfillColor fillColor) {\n"
    "  return fillColor;\n"
    "}\n"
    "\n"
    "////////////////////////////////////////////////////////////////\n"
    "/**\n"
    " * Non-interpolated shader outputs.\n"
    " */\n"
    "flat out uint fPrimitiveId;  // the id of the primitive\n"
    "\n"
    "// the output color of the primitive\n"
    "#if usePerVertColor == 1\n"
    "out vec4 fColor;\n"
    "#else\n"
    "flat out vec4 fColor;\n"
    "#endif\n"
    "\n"
    "void main() {\n"
    "  gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);\n"
    "\n"
    "  #if usePerVertColor == 1 || useUfillColor\n"
    "  fColor = getfillColor(fillColor);\n"
    "  #elif useUniformBuffer == 1\n"
    "  fColor = getfillColor(polyData.fillColor);\n"
    "  #endif\n"
    "\n"
    "  // ids from queries go from 0 to numrows-1, but since we're storing\n"
    "  // the ids as unsigned ints, and there isn't a way to specify the\n"
    "  // clear value for secondary buffers, we need to account for that\n"
    "  // offset here\n"
    "  #if useUid == 1\n"
    "  fPrimitiveId = id + 1;\n"
    "  #elif useUniformBuffer == 1\n"
    "  fPrimitiveId = polyData.id + 1;\n"
    "  #endif\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // POLYTEMPLATE_VERT_H_