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
    "rangeType_<name> getQuantizeScale_<name>(in domainType_<name> category) {\n"
    "    double diff = double(category - uDomains_<name>[0]);\n"
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
