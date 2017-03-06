#ifndef POLYTEMPLATE_FRAG_H_
#define POLYTEMPLATE_FRAG_H_

#include <string>

namespace QueryRenderer {
struct PolyTemplate_Frag {
  static const std::string source;
};

const std::string PolyTemplate_Frag::source =
    "#version 410 core\n"
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
    "uniform int uDataId;\n"
    "flat in inTid fPrimitiveId;\n"
    "\n"
    "#define usePerVertColor <usePerVertColor>\n"
    "#if usePerVertColor == 1\n"
    "in vec4 fColor;\n"
    "#else\n"
    "flat in vec4 fColor;\n"
    "#endif\n"
    "\n"
    "void main() {\n"
    "    // TODO(croot): do some kind of temporary anti-aliasing?\n"
    "    color = fColor;\n"
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
    "    tableId = sign(uTableId) * ((abs(uTableId + 1) << 5) | (uDataId + 1));\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // POLYTEMPLATE_FRAG_H_
