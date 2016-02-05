#ifndef POINTTEMPLATE_VERT_H_
#define POINTTEMPLATE_VERT_H_

#include <string>

namespace QueryRenderer {
struct PointTemplate_Vert {
  static const std::string source;
};

const std::string PointTemplate_Vert::source =
    "// VERTEX SHADER\n"
    "\n"
    "#version 450 core\n"
    "\n"
    "#define useUx <useUx>\n"
    "#define inTx <inTxType>\n"
    "#define outTx <outTxType>\n"
    "\n"
    "#define useUy <useUy>\n"
    "#define inTy <inTyType>\n"
    "#define outTy <outTyType>\n"
    "\n"
    "// #define useUz <useUz>\n"
    "// #define inTz float\n"
    "// #define outTz float\n"
    "\n"
    "#define useUsize <useUsize>\n"
    "#define inTsize <inTsizeType>\n"
    "#define outTsize <outTsizeType>\n"
    "\n"
    "#define useUfillColor <useUfillColor>\n"
    "#define inTfillColor <inTfillColorType>\n"
    "#define outTfillColor <outTfillColorType>\n"
    "\n"
    "#define useUid <useUid>\n"
    "\n"
    "#define useKey <useKey>\n"
    "#define useUkey <useUkey>\n"
    "\n"
    "#if useUx == 1\n"
    "uniform inTx x;\n"
    "#else\n"
    "in inTx x;\n"
    "#endif\n"
    "\n"
    "#if useUy == 1\n"
    "uniform inTy y;\n"
    "#else\n"
    "in inTy y;\n"
    "#endif\n"
    "\n"
    "// #if useUz == 1\n"
    "// uniform inTz z;\n"
    "// #else\n"
    "// in inTz z;\n"
    "// #endif\n"
    "\n"
    "#if useUsize == 1\n"
    "uniform inTsize size;\n"
    "#else\n"
    "in inTsize size;\n"
    "#endif\n"
    "\n"
    "#if useUfillColor == 1\n"
    "uniform inTfillColor fillColor;\n"
    "#else\n"
    "in inTfillColor fillColor;\n"
    "#endif\n"
    "\n"
    "#if useUid == 1\n"
    "uniform uint id;\n"
    "#else\n"
    "in uint id;\n"
    "#endif\n"
    "\n"
    "#if useKey == 1\n"
    "// TODO(croot): use the NV_vertex_attrib_integer_64bit extension to use an int64 for the key and invalidKey.\n"
    "#if useUkey == 1\n"
    "uniform int key;\n"
    "#else\n"
    "in int key;\n"
    "#endif\n"
    "uniform int invalidKey;\n"
    "#endif\n"
    "\n"
    "outTx getx(in inTx x) {\n"
    "return x;\n"
    "}\n"
    "\n"
    "outTy gety(in inTy y) {\n"
    "return y;\n"
    "}\n"
    "\n"
    "// outTz getz(in inTz z) {\n"
    "//     return z;\n"
    "// }\n"
    "\n"
    "outTsize getsize(in inTsize size) {\n"
    "return size;\n"
    "}\n"
    "\n"
    "outTfillColor getfillColor(in inTfillColor fillColor) {\n"
    "return fillColor;\n"
    "}\n"
    "\n"
    "////////////////////////////////////////////////////////////////\n"
    "/**\n"
    "* Non-interpolated shader outputs.\n"
    "*/\n"
    "flat out uint fPrimitiveId;  // the id of the primitive\n"
    "flat out vec4 fColor;        // the output color of the primitive\n"
    "flat out float fPointSize;   // the point size of the vertex\n"
    "\n"
    "void main() {\n"
    "#if useKey == 1\n"
    "if (key != invalidKey) {\n"
    "gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);\n"
    "float sz = getsize(size);\n"
    "fPointSize = sz;\n"
    "gl_PointSize = sz;\n"
    "\n"
    "fColor = getfillColor(fillColor);\n"
    "} else {\n"
    "gl_Position = vec4(0, 0, 0, 0);\n"
    "fPointSize = 0.0;\n"
    "gl_PointSize = 0.0;\n"
    "fColor = vec4(0, 0, 0, 0);\n"
    "}\n"
    "#else\n"
    "gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);\n"
    "float sz = getsize(size);\n"
    "fPointSize = sz;\n"
    "gl_PointSize = sz;\n"
    "\n"
    "fColor = getfillColor(fillColor);\n"
    "#endif\n"
    "\n"
    "// ids from queries go from 0 to numrows-1, but since we're storing\n"
    "// the ids as unsigned ints, and there isn't a way to specify the\n"
    "// clear value for secondary buffers, we need to account for that\n"
    "// offset here\n"
    "fPrimitiveId = id + 1;\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // POINTTEMPLATE_VERT_H_
