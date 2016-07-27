#ifndef ACCUMULATORTX_FINDSTDDEV_H_
#define ACCUMULATORTX_FINDSTDDEV_H_

#include <string>

namespace QueryRenderer {
struct AccumulatorTx_FindStdDev_frag {
  static const std::string source;
};

const std::string AccumulatorTx_FindStdDev_frag::source =
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
    // NOTE: float conversions can be much faster than doubles, so there's a tradeoff between speed and accuracy here
    "        float mean = float(imageLoad(inExtents, ivec2(2, 0)).r) / float(imageLoad(inExtents, ivec2(3, 0)).r);\n"
    // "        double mean = double(imageLoad(inExtents, ivec2(2, 0)).r) / double(imageLoad(inExtents, ivec2(3,
    // 0)).r);\n"

    // NOTE: the following code rounds the double off. I'm hoping the error introduced will be reasonable enough to
    // calculate
    // the std dev in density accumulations. The use case suggests it doesn't have to be exact.
    "        float sqrDiff = pow(float(cnt) - mean, 2.0);\n"
    // "        double sqrDiff = pow(double(cnt) - mean, 2.0);\n"
    "        uint roundSqrDiff = uint(round(sqrDiff));\n"
    "        imageAtomicAdd(inExtents, ivec2(4, 0), roundSqrDiff);\n"
    "    }\n"
    "\n"
    "    color = vec4(0,0,0,0);\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // ACCUMULATORTX_FINDSTDDEV_H_
