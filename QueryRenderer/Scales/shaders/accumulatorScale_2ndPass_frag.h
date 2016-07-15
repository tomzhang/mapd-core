#ifndef ACCUMULATORSCALE_2NDPASS_FRAG_H_
#define ACCUMULATORSCALE_2NDPASS_FRAG_H_

#include <string>

namespace QueryRenderer {
struct AccumulatorScale_2ndPass_frag {
  static const std::string source;
};

const std::string AccumulatorScale_2ndPass_frag::source =
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    // "layout(location = 1) out uint id;\n"
    "\n"
    "\n"
    "#define numAccumColors_<name> <numAccumColors>\n"
    "uniform vec4 inColors[numAccumColors_<name>];\n"
    "\n"
    "#define numAccumTextures_<name> <numAccumTextures>\n"
    "layout(r32ui, binding=0) uniform readonly uimage2D inTxPixelCounter[numAccumTextures_<name>];\n"
    "layout(r32ui, binding=1) uniform readonly uimage2DArray inTxArrayPixelCounter;\n"
    "\n"
    "subroutine uint GetAccumulatedCnt(uint);\n"
    "subroutine uniform GetAccumulatedCnt getAccumulatedCnt;\n"
    "subroutine(GetAccumulatedCnt) uint getTxAccumulatedCnt(uint idx) {\n"
    "   return imageLoad(inTxPixelCounter[idx], ivec2(gl_FragCoord.xy)).r;\n"
    "}\n"
    "\n"
    "subroutine(GetAccumulatedCnt) uint getTxArrayAccumulatedCnt(uint idx) {\n"
    "   return imageLoad(inTxArrayPixelCounter, ivec3(gl_FragCoord.xy, idx)).r;\n"
    "}\n"
    "\n"
    "subroutine vec4 AccumulatedColorFunc();\n"
    "subroutine uniform AccumulatedColorFunc getAccumulatedColor;\n"
    "\n"
    "subroutine(AccumulatedColorFunc) vec4 getMinAccumulatedColor() {\n"
    "    uint idx = 0;\n"
    "    uint cnt;\n"
    "    vec4 finalColor = vec4(0,0,0,0);\n"
    "\n"
    "    uint cnt1, cnt2;\n"
    "    int minIdx = -1;\n"
    "    uint minCnt = 10000000;\n"
    "    for (int i = 0; i < numAccumColors_<name>; i+=2) {\n"
    "        idx = i / 2;\n"
    "\n"
    "        if (idx < numAccumTextures_<name>) {\n"
    "            cnt = getAccumulatedCnt(idx);\n"
    "            cnt1 = cnt & uint(0x000000FF);\n"
    "            if (cnt1 > 0 && cnt1 < minCnt) {\n"
    "                minCnt = cnt1;\n"
    "                minIdx = i;\n"
    "            }\n"
    "            cnt2 = (cnt & uint(0x00FF0000)) >> 16;\n"
    "            if (cnt2 > 0 && cnt2 < minCnt) {\n"
    "                minCnt = cnt2;\n"
    "                minIdx = i + 1;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    if (minIdx >= 0) {\n"
    "        finalColor = inColors[minIdx];\n"
    "    }\n"
    "    return finalColor;\n"
    "}\n"
    "\n"
    "subroutine(AccumulatedColorFunc) vec4 getMaxAccumulatedColor() {\n"
    "    uint idx = 0;\n"
    "    uint cnt;\n"
    "    vec4 finalColor = vec4(0,0,0,0);\n"
    "\n"
    "    uint cnt1, cnt2;\n"
    "    int maxIdx = -1;\n"
    "    uint maxCnt = 0;\n"
    "    for (int i = 0; i < numAccumColors_<name>; i+=2) {\n"
    "        idx = i / 2;\n"
    "\n"
    "        if (idx < numAccumTextures_<name>) {\n"
    "            cnt = getAccumulatedCnt(idx);\n"
    "            cnt1 = cnt & uint(0x000000FF);\n"
    "            if (cnt1 > 0 && cnt1 > maxCnt) {\n"
    "                maxCnt = cnt1;\n"
    "                maxIdx = i;\n"
    "            }\n"
    "            cnt2 = (cnt & uint(0x00FF0000)) >> 16;\n"
    "            if (cnt2 > 0 && cnt2 > maxCnt) {\n"
    "                maxCnt = cnt2;\n"
    "                maxIdx = i + 1;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    if (maxIdx >= 0) {\n"
    "        finalColor = inColors[maxIdx];\n"
    "    }\n"
    "    return finalColor;\n"
    "}\n"
    "\n"
    "subroutine(AccumulatedColorFunc) vec4 getBlendAccumulatedColor() {\n"
    "    uint idx = 0;\n"
    "    uint cnt;\n"
    "    vec4 finalColor = vec4(0,0,0,0);\n"
    "\n"
    "    uint cnts[numAccumColors_<name> + (numAccumColors_<name> % 2)];\n"
    "    uint totalCnt = 0;\n"
    "    for (uint i = 0; i < numAccumColors_<name>; i+=2) {\n"
    "        idx = i / 2;\n"
    "\n"
    "        if (idx < numAccumTextures_<name>) {\n"
    "            cnt = getAccumulatedCnt(idx);\n"
    "            cnts[i] = cnt & uint(0x000000FF);\n"
    "            cnts[i+1] = (cnt & uint(0x00FF0000)) >> 16;\n"
    "            totalCnt += cnts[i] + cnts[i+1];\n"
    "        }\n"
    "    }\n"
    "\n"
    "    float sum = float(totalCnt);\n"
    "    for (uint i = 0; i < numAccumColors_<name>; i++) {\n"
    "        finalColor += (float(cnts[i]) / sum) * inColors[i];\n"
    "    }\n"
    "    return finalColor;\n"
    "}\n"
    "\n"
    "uniform uint minDensity;\n"
    "uniform uint maxDensity;\n"
    "uniform usampler2D densityExtents;\n"
    "\n"
    "subroutine uint GetExtentFunc();\n"
    "subroutine uniform GetExtentFunc getMinDensity;\n"
    "subroutine uniform GetExtentFunc getMaxDensity;\n"
    "\n"
    "subroutine(GetExtentFunc) uint getUniformMinDensity() {\n"
    "    return minDensity;\n"
    "}\n"
    "\n"
    "subroutine(GetExtentFunc) uint getTextureMinDensity() {\n"
    "    return texelFetch(densityExtents, ivec2(0, 0), 0).r;\n"
    "}\n"
    "\n"
    "subroutine(GetExtentFunc) uint getUniformMaxDensity() {\n"
    "    return maxDensity;\n"
    "}\n"
    "\n"
    "subroutine(GetExtentFunc) uint getTextureMaxDensity() {\n"
    "    return texelFetch(densityExtents, ivec2(1, 0), 0).r;\n"
    "}\n"
    "\n"
    "vec4 getDensityColor(double pct) {\n"
    "    return float(pct)*inColors[0];\n"
    "}\n"
    "\n"
    "subroutine(AccumulatedColorFunc) vec4 getDensityAccumulatedColor() {\n"
    "    uint totalCnt = 0;\n"
    "\n"
    "    // there should only be 1 counter texture for density accumulation\n"
    "    totalCnt = getAccumulatedCnt(0);\n"
    "    if (totalCnt == 0) {\n"
    "        return vec4(0,0,0,0);\n"
    "    }\n"
    "    uint myMinDensity = getMinDensity();\n"
    "    uint myMaxDensity = getMaxDensity();\n"
    "    domainType_<name>_DENSITY pct = domainType_<name>_DENSITY(totalCnt - myMinDensity) / \n"
    "domainType_<name>_DENSITY(myMaxDensity - myMinDensity);\n"
    "    return getDensityColor(pct);\n"
    "}\n"
    //"uniform usampler2DMS idSampler;\n"
    "\n"
    "void main() {\n"
    "    color = getAccumulatedColor();\n"
    //"    id = texelFetch(idSampler, ivec2(gl_FragCoord.xy), 0).r;\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // ACCUMULATORSCALE_2NDPASS_FRAG_H_
