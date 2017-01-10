#ifndef QUERYRENDERER_GLX_ACCUMULATORFRAG_H_
#define QUERYRENDERER_GLX_ACCUMULATORFRAG_H_

#include <string>

namespace QueryRenderer {
namespace Impl {
namespace GLX {

struct Accumulator_frag {
  static const std::string source;
};

const std::string Accumulator_frag::source =
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(location = 1) out uint idA;\n"
    "layout(location = 2) out uint idB;\n"
    "layout(location = 3) out uint tableId;\n"
    "\n"
    "uniform usampler2DArray accumulatorSampler;\n"
    "layout(r32ui, binding=0) uniform coherent uimage2DArray inTxPixelCounter;\n"
    "\n"
    "uniform int numAccumTextures;\n"
    "\n"
    "void main() {\n"
    "  uint cnt, cnt1, cnt2, shiftIdx, shift;\n"
    "  for (uint i=0; i<numAccumTextures; ++i) {\n"
    "    cnt = texelFetch(accumulatorSampler, ivec3(gl_FragCoord.xy, i), 0).r;\n"
    "\n"
    "    imageAtomicAdd(inTxPixelCounter, ivec3(gl_FragCoord.xy, i), cnt);\n"
    "  }\n"
    "}\n";

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_GLX_ACCUMULATORFRAG_H_
