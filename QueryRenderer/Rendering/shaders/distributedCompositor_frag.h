#ifndef QUERYRENDERER_GLX_DISTRIBUTEDCOMPOSITORFRAG_H_
#define QUERYRENDERER_GLX_DISTRIBUTEDCOMPOSITORFRAG_H_

#include <string>

namespace QueryRenderer {

struct DistributedCompositor_frag {
  static const std::string source;
};

const std::string DistributedCompositor_frag::source =
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(location = 1) out uint idA;\n"
    "layout(location = 2) out uint idB;\n"
    "layout(location = 3) out int tableId;\n"
    "\n"
    "uniform sampler2D rgbaSampler;\n"
    "uniform usampler2D id1ASampler;\n"
    "uniform usampler2D id1BSampler;\n"
    "uniform isampler2D id2Sampler;\n"
    "\n"
    "uniform bool useIdBSampler;\n"
    "\n"
    "void main() {\n"
    "    vec4 srcColor = texelFetch(rgbaSampler, ivec2(gl_FragCoord.xy), 0);\n"
    "    if (srcColor.a == 0.0) {\n"
    "        discard;\n"
    "    }\n"
    "    color = srcColor;\n"
    "    idA = texelFetch(id1ASampler, ivec2(gl_FragCoord.xy), 0).r;\n"
    "    if (useIdBSampler) {\n"
    "        idB = texelFetch(id1BSampler, ivec2(gl_FragCoord.xy), 0).r;\n"
    "    }\n"
    "    tableId = texelFetch(id2Sampler, ivec2(gl_FragCoord.xy), 0).r;\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_GLX_DISTRIBUTEDCOMPOSITORFRAG_H_
