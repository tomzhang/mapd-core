#ifndef QUANTIZESCALETEMPLATE_VERT_H_
#define QUANTIZESCALETEMPLATE_VERT_H_

#include <string>

namespace QueryRenderer {
struct QuantizeScaleTemplate_vert {
  static const std::string source;
};

const std::string QuantizeScaleTemplate_vert::source =
    "#define domainType_<name> <domainType>\n"
    "#define rangeType_<name> <rangeType>\n"
    "\n"
    "#define numRanges_<name> <numRanges>\n"
    "\n"
    "uniform double quantizeDiff;\n"
    "uniform domainType_<name> uDomains_<name>[2];\n"
    "uniform rangeType_<name> uRanges_<name>[numRanges_<name>];\n"
    "\n"
    "#define doAccum_QuantizeScale_<name> <doAccum>\n"
    "\n"
    "#if doAccum_QuantizeScale_<name> == 1\n"
    "flat out int accumIdx;   // the ordinal domain index for accumulations\n"
    "#endif\n"
    "\n"
    // TODO(croot): right now, if the quantize scale is a density
    // accumulation, we still find the index via the following math
    // unnecessarily. The better approach is to make a subroutine
    // or an additional #if/#else ... but that will take a little
    // bit of work. Right now (as of 07/15/16) I'm changing
    // accumulatorScale_1stPass_frag.h to automatically set the
    // accumulator texture index to 0 rather than using 'accumTx'
    "rangeType_<name> getQuantizeScale_<name>(in domainType_<name> category) {\n"
    "    double diff;\n"
    "    if (uDomains_<name>[0] >= 0) {\n"
    "        diff = min(double(category), double(category - uDomains_<name>[0]));\n"
    "    } else {\n"
    "        diff = max(double(category), double(category - uDomains_<name>[0]));\n"
    "    }\n"
    "    int idx = int(max(min(trunc(diff / quantizeDiff), double(numRanges_<name>-1)), "
    "double(0)));\n"
    "\n"
    "    #if doAccum_QuantizeScale_<name> == 1\n"
    "    accumIdx = idx;\n"
    "    return vec4(0,0,0,0);\n"
    "    #else\n"
    "    return uRanges_<name>[idx];\n"
    "    #endif\n"
    "}\n";
}  // namespace QueryRenderer

#endif  // QUANTIZESCALETEMPLATE_VERT_H_
