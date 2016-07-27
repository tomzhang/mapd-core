#ifndef ACCUMULATORSCALE_1STPASS_FRAG_H_
#define ACCUMULATORSCALE_1STPASS_FRAG_H_

#include <string>

namespace QueryRenderer {
struct AccumulatorScale_1stPass_frag {
  static const std::string source;
};

const std::string AccumulatorScale_1stPass_frag::source =
    "#define numAccumTextures_<name> <numAccumTextures>\n"
    "layout(r32ui, binding=0) uniform coherent uimage2D inTxPixelCounter[numAccumTextures_<name>];\n"
    "flat in int accumIdx;\n"
    "\n"
    "subroutine void AccumulatorFunc();\n"
    "subroutine uniform AccumulatorFunc accumulate;\n"
    "\n"
    "subroutine(AccumulatorFunc) void minMaxBlendAccumulate() {\n"
    "    uint currIdx = accumIdx / 2;\n"
    "    uint shiftIdx = accumIdx % uint(2);\n"
    "    uint shift = shiftIdx * 16;\n"
    "\n"
    "    uint incr = 1 << shift;\n"
    "    imageAtomicAdd(inTxPixelCounter[currIdx], ivec2(gl_FragCoord.xy), incr);\n"
    "}\n"
    "\n"
    "subroutine(AccumulatorFunc) void densityAccumulate() {\n"
    "    // NOTE: accumIdx should always be 0 in the density case\n"
    "    imageAtomicAdd(inTxPixelCounter[0], ivec2(gl_FragCoord.xy), 1);\n"
    "}\n"
    "\n"
    "vec4 getFragColor() {\n"
    "    accumulate();\n"
    "    return vec4(0,0,0,0);\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // ACCUMULATORSCALE_1STPASS_FRAG_H_
