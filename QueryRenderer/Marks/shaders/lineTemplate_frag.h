#ifndef LINETEMPLATE_FRAG_H_
#define LINETEMPLATE_FRAG_H_

#include <string>

namespace QueryRenderer {
struct LineTemplate_Frag {
  static const std::string source;
};

const std::string LineTemplate_Frag::source =
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 finalColor;\n"
    "layout(location = 1) out uint finalId;\n"
    "\n"
    "#define usePerVertId <usePerVertId>\n"
    "#define useUid <useUid>\n"
    "#if usePerVertId == 1\n"
    "flat in uint fId;\n"
    "#elif useUid == 1\n"
    "uniform uint id;\n"
    "#endif\n"
    "\n"
    "#define usePerVertColor <usePerVertColor>\n"
    "#define useUstrokeColor <useUstrokeColor>\n"
    "#define inTstrokeColor <inTstrokeColorType>\n"
    "#define outTstrokeColor <outTstrokeColorType>\n"
    "#if usePerVertColor == 1\n"
    "in vec4 fColor;\n"
    "#elif useUstrokeColor == 1\n"
    "uniform inTstrokeColor strokeColor;\n"
    "#endif\n"
    "\n"
    "#define useUlineJoin <useUlineJoin>\n"
    "#if useUlineJoin == 1\n"
    "uniform int lineJoin;\n"
    "#endif\n"
    "\n"
    "#define useUniformBuffer <useUniformBuffer>\n"
    "#if useUniformBuffer == 1\n"
    "<lineData>\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 0\n"
    "outTstrokeColor getstrokeColor(in inTstrokeColor strokeColor) {\n"
    "  return strokeColor;\n"
    "}\n"
    "#endif\n"
    "\n"
    "in vec2 fNormDistCoords;\n"
    "flat in int fIsCorner;\n"
    "\n"
    "void main() {\n"
    "  if (fIsCorner == 1) {\n"
    "#if useUlineJoin == 1\n"
    "    if ((lineJoin == 1 && length(fNormDistCoords) > 1) ||\n"
    "        (lineJoin != 1 && fNormDistCoords[0] > 1)) {\n"
    "#else // should be in uniform buffer\n"
    "    if ((lineData.<lineJoin> == 1 && length(fNormDistCoords) > 1) ||\n"
    "        (lineData.<lineJoin> != 1 && fNormDistCoords[0] > 1)) {\n"
    "#endif\n"
    "      discard;\n"
    "    }\n"
    "  }\n"
    "\n"
    "#if usePerVertId == 1\n"
    "  finalId = fId + 1;\n"
    "#elif useUid == 1\n"
    "  finalId = id + 1;\n"
    "#elif useUniformBuffer == 1\n"
    "  finalId = lineData.<id> + 1;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "  finalColor = fColor;\n"
    "#elif useUstrokeColor == 1\n"
    "  finalColor = getstrokeColor(strokeColor);\n"
    "#elif useUniformBuffer == 1\n"
    "  finalColor = getstrokeColor(lineData.<strokeColor>);\n"
    "#endif\n"
    "\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // LINETEMPLATE_FRAG_H_
