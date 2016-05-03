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
    "uniform domainType_<name> quantizeDiff;\n"
    "uniform domainType_<name> uDomains_<name>[2];\n"
    "uniform rangeType_<name> uRanges_<name>[numRanges_<name>];\n"
    "\n"
    "rangeType_<name> getQuantizeScale_<name>(in domainType_<name> category) {\n"
    "    domainType_<name> diff = category - uDomains_<name>[0];\n"
    "    int idx = int(max(min(trunc(diff / quantizeDiff), domainType_<name>(numRanges_<name>-1)), "
    "domainType_<name>(0)));\n"
    "\n"
    "    rangeType_<name> val = uRanges_<name>[idx];\n"
    "\n"
    "    return val;\n"
    "}\n";
}  // namespace QueryRenderer

#endif  // QUANTIZESCALETEMPLATE_VERT_H_
