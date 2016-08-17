#ifndef QUANTITATIVESCALETEMPLATE_VERT_H_
#define QUANTITATIVESCALETEMPLATE_VERT_H_

#include <string>

namespace QueryRenderer {
struct QuantitativeScaleTemplate_vert {
  static const std::string source;
};

const std::string QuantitativeScaleTemplate_vert::source =
    "#define domainType_<name> <domainType>\n"
    "#define rangeType_<name> <rangeType>\n"
    "\n"
    "#define numDomains_<name> <numDomains>\n"
    "#define numRanges_<name> <numRanges>\n"
    "#define useClamp_<name> <useClamp>\n"
    "\n"
    "uniform domainType_<name> uDomains_<name>[numDomains_<name>];\n"
    "uniform domainType_<name> nullDomainVal_<name>;\n"
    "uniform rangeType_<name> uRanges_<name>[numRanges_<name>];\n"
    "uniform rangeType_<name> nullRangeVal_<name>;\n"
    "\n"
    "subroutine bool IsNullValSubroutine_<name>(in domainType_<name>);\n"
    "subroutine uniform IsNullValSubroutine_<name> isNullValFunc_<name>;\n"
    "subroutine(IsNullValSubroutine_<name>) bool isNullValPassThru_<name>(in domainType_<name> val) {\n"
    "    return false;\n"
    "}\n"
    "\n"
    "subroutine(IsNullValSubroutine_<name>) bool isNullVal_<name>(in domainType_<name> val) {\n"
    "    if (val == nullDomainVal_<name>) {\n"
    "        return true;\n"
    "    }\n"
    "    return false;\n"
    "}\n"
    "\n"
    "#define doAccum_QuantitativeScale_<name> <doAccum>\n"
    "\n"
    "#if doAccum_QuantitativeScale_<name> == 1\n"
    "flat out int accumIdx;   // the ordinal domain index for accumulations\n"
    "#endif\n"
    "\n"
    "subroutine domainType_<name> QuantTransformFunc_<name>(in domainType_<name>);\n"
    "subroutine uniform QuantTransformFunc_<name> quantTransform_<name>;\n"
    "\n"
    "subroutine(QuantTransformFunc_<name>) domainType_<name> passThruTransform_<name>(in domainType_<name> val) {\n"
    "    return val;\n"
    "}\n"
    "\n"
    "subroutine(QuantTransformFunc_<name>) domainType_<name> logTransform_<name>(in domainType_<name> val) {\n"
    "    return domainType_<name>(log(float(val)));\n"
    "}\n"
    "\n"
    "uniform float uExponent_<name>;\n"
    "subroutine(QuantTransformFunc_<name>) domainType_<name> powTransform_<name>(in domainType_<name> val) {\n"
    "    return domainType_<name>(pow(float(val), uExponent_<name>));\n"
    "}\n"
    "\n"
    "subroutine(QuantTransformFunc_<name>) domainType_<name> sqrtTransform_<name>(in domainType_<name> val) {\n"
    "    return domainType_<name>(sqrt(float(val)));\n"
    "}\n"
    "\n"
    "rangeType_<name> getQuantitativeScale_<name>(in domainType_<name> domainVal) {\n"
    "    domainType_<name> val1;\n"
    "    domainType_<name> val2;\n"
    "    int idx1, idx2;\n"
    "\n"
    "    #if doAccum_QuantitativeScale_<name> == 1\n"
    "        accumIdx = 0;\n"
    "        return uRanges_<name>[accumIdx];\n"
    "    #else\n"
    "      if (isNullValFunc_<name>(domainVal)) {\n"
    "        return nullRangeVal_<name>;\n"
    "      }\n"
    "\n"
    "      domainType_<name> transformedVal = quantTransform_<name>(domainVal);\n"
    "      #if numDomains_<name> == 1 || numRanges_<name> == 1\n"
    "        idx1 = 0;\n"
    "        idx2 = 0;\n"
    "        val1 = uDomains_<name>[0];\n"
    "        val2 = uDomains_<name>[0];\n"
    "      #elif numDomains_<name> == 2 || numRanges_<name> == 2\n"
    "        idx1 = 0;\n"
    "        idx2 = 1;\n"
    "        val1 = uDomains_<name>[0];\n"
    "        val2 = uDomains_<name>[1];\n"
    "      #else\n"
    "        int startIdx = 0;\n"
    "        int endIdx = numDomains_<name> - 1;\n"
    "        domainType_<name> midVal;\n"
    "\n"
    "        if (transformedVal == uDomains_<name>[startIdx]) {\n"
    "            idx1 = startIdx;\n"
    "            idx2 = startIdx+1;\n"
    "        } else if (transformedVal == uDomains_<name>[endIdx]) {\n"
    "            idx1 = endIdx-1;\n"
    "            idx2 = endIdx;\n"
    "        } else {\n"
    "            while (true) {\n"
    "                int midIdx = startIdx + (endIdx - startIdx) / 2;\n"
    "                if (midIdx == startIdx) {\n"
    "                    idx1 = startIdx;\n"
    "                    idx2 = endIdx;\n"
    "                    break;\n"
    "                } else {\n"
    "                    midVal = uDomains_<name>[midIdx];\n"
    "                    if (transformedVal == midVal) {\n"
    "                        idx1 = midIdx;\n"
    "                        idx2 = midIdx+1;\n"
    "                        break;\n"
    "                    } else if (transformedVal > midVal) {\n"
    "                        startIdx = midIdx;\n"
    "                    } else {\n"
    "                        endIdx = midIdx;\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "\n"
    "        val1 = uDomains_<name>[idx1];\n"
    "        val2 = uDomains_<name>[idx2];\n"
    "      #endif\n"
    "\n"
    "      float t = (float(transformedVal) - float(val1)) / (float(val2) - float(val1));\n"
    "\n"
    "      #if useClamp_<name> == 1\n"
    "        t = clamp(t, 0.0, 1.0);\n"
    "      #endif\n"
    "\n"
    "      return mix(uRanges_<name>[idx1], uRanges_<name>[idx2], t);\n"
    "    #endif // doAccum_QuantitativeScale_<name>\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // QUANTITATIVESCALETEMPLATE_VERT_H_
