#ifndef ACCUMULATORTX_FINDEXTENTS_H_
#define ACCUMULATORTX_FINDEXTENTS_H_

#include <string>

namespace QueryRenderer {
struct AccumulatorTx_FindExtents_frag {
  static const std::string source;
};

const std::string AccumulatorTx_FindExtents_frag::source =
    "// FRAGMENT SHADER\n"
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "\n"
    "layout(r32ui, binding=0) uniform coherent uimage2D inExtents;\n"
    "\n"
    "// NOTE: this should only be called in the density accumulation\n"
    "// case, and thus there should only be 1 counter texture\n"
    "uniform usampler2D inTxPixelCounter;\n"
    "uniform usampler2DArray inTxArrayPixelCounter;\n"
    "\n"
    "subroutine uint GetAccumulatedCnt();\n"
    "subroutine uniform GetAccumulatedCnt getAccumulatedCnt;\n"
    "subroutine(GetAccumulatedCnt) uint getTxAccumulatedCnt() {\n"
    "   return texelFetch(inTxPixelCounter, ivec2(gl_FragCoord.xy), 0).r;\n"
    "}\n"
    "\n"
    "subroutine(GetAccumulatedCnt) uint getTxArrayAccumulatedCnt() {\n"
    "   return texelFetch(inTxArrayPixelCounter, ivec3(gl_FragCoord.xy, 0), 0).r;\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    uint cnt = getAccumulatedCnt();\n"
    "    if (cnt > 0) {\n"
    "        imageAtomicMin(inExtents, ivec2(0, 0), cnt);\n"
    "        imageAtomicMax(inExtents, ivec2(1, 0), cnt);\n"
    "    }\n"
    "\n"
    "    color = vec4(0,0,0,0);\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // ACCUMULATORTX_FINDEXTENTS_H_
