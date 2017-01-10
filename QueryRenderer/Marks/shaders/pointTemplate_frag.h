#ifndef POINTTEMPLATE_FRAG_H_
#define POINTTEMPLATE_FRAG_H_

#include <string>

namespace QueryRenderer {
struct PointTemplate_Frag {
  static const std::string source;
};

const std::string PointTemplate_Frag::source =
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(location = 1) out uint idA;\n"
    "layout(location = 2) out uint idB;\n"
    "layout(location = 3) out int tableId;\n"
    "\n"
    "#define inTid <inTidType>\n"
    "#define inTidEnum <inTidEnum>\n"
    "\n"
    "uniform int uTableId;\n"
    "flat in inTid fPrimitiveId;\n"
    "flat in vec4 fColor;\n"
    "flat in float fPointSize;\n"
    "\n"
    "vec4 getFragColor() {\n"
    "    return fColor;\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    float dist = distance(gl_PointCoord, vec2(0.5));\n"
    "\n"
    "    if (fPointSize < 3.0) {\n"
    "        if (dist > 0.5) {\n"
    "            discard;\n"
    "        }\n"
    "        color = getFragColor();\n"
    "    } else {\n"
    "        float delta = fwidth(dist);\n"
    "        float alpha = 1.0 - smoothstep(0.5-delta, 0.5, dist);\n"
    "        if (alpha == 0.0) {\n"
    "            discard;\n"
    "        }\n"
    "\n"
    "        vec4 fragColor = getFragColor();\n"
    "        color = vec4(fragColor.xyz, fragColor.w*alpha);\n"
    "    }\n"
    "#if inTidEnum == UNSIGNED_INT64_NV\n"
    "    uvec2 vals = unpackUint2x32(fPrimitiveId);\n"
    "    idA = vals[0];\n"
    "    idB = vals[1];\n"
    "#elif inTidEnum == INT64_NV\n"
    "    ivec2 vals = unpackInt2x32(fPrimitiveId);\n"
    "    idA = uint(vals[0]);\n"
    "    idB = uint(vals[1]);\n"
    "#else\n"
    "    idA = fPrimitiveId;\n"
    "    idB = 0;\n"
    "#endif\n"
    "    tableId = uTableId + 1;\n"
    "}\n";
}  // namespace QueryRenderer

#endif  // POINTTEMPLATE_FRAG_H_
