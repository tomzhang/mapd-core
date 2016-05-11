#ifndef LINETEMPLATE_GEOM_H_
#define LINETEMPLATE_GEOM_H_

#include <string>

namespace QueryRenderer {
struct LineTemplate_Geom {
  static const std::string source;
};

const std::string LineTemplate_Geom::source =
    "// GEOMETRY SHADER\n"
    "\n"
    "#version 450 core\n"
    "\n"
    "layout(lines_adjacency) in;\n"
    "layout(triangle_strip, max_vertices = 10) out;\n"
    "\n"
    "struct Viewport\n"
    "{\n"
    "  int x;\n"
    "  int y;\n"
    "  int width;\n"
    "  int height;\n"
    "};\n"
    "uniform Viewport viewport;\n"
    "\n"
    "#define usePerVertId <usePerVertId>\n"
    "#define useUid <useUid>\n"
    "#if usePerVertId == 1\n"
    "in uint gId[];\n"
    "#elif useUid == 1\n"
    "uniform uint id;\n"
    "#endif\n"
    "\n"
    "#define usePerVertColor <usePerVertColor>\n"
    "#if usePerVertColor == 1\n"
    "in vec4 gColor[];\n"
    "#endif\n"
    "\n"
    "#define usePerVertWidth <usePerVertWidth>\n"
    "#define useUstrokeWidth <useUstrokeWidth>\n"
    "#define inTstrokeWidth <inTstrokeWidthType>\n"
    "#define outTstrokeWidth <outTstrokeWidthType>\n"
    "#if usePerVertWidth == 1\n"
    "in float gWidth[];\n"
    "#elif useUstrokeWidth == 1\n"
    "uniform inTstrokeWidth strokeWidth;\n"
    "#endif\n"
    "\n"
    "#define useUlineJoin <useUlineJoin>\n"
    "#if useUlineJoin == 1\n"
    "uniform int lineJoin;\n"
    "#endif\n"
    "\n"
    "#define inTmiterLimit <inTmiterLimitType>\n"
    "#define outTmiterLimit <outTmiterLimitType>\n"
    "#define useUmiterLimit <useUmiterLimit>\n"
    "#if useUmiterLimit == 1\n"
    "uniform inTmiterLimit miterLimit;\n"
    "#endif\n"
    "\n"
    "#define useUniformBuffer <useUniformBuffer>\n"
    "#if useUniformBuffer == 1\n"
    "<lineData>\n"
    "#endif\n"
    "\n"
    "#if usePerVertWidth == 0\n"
    "outTstrokeWidth getstrokeWidth(in inTstrokeWidth strokeWidth)  {\n"
    "  return strokeWidth;\n"
    "}\n"
    "#endif\n"
    "\n"
    "outTmiterLimit getmiterLimit(in inTmiterLimit miterLimit) {\n"
    "  return miterLimit;\n"
    "}\n"
    "\n"
    "out vec2 fNormDistCoords;\n"
    "flat out int fIsCorner;\n"
    "\n"
    "#if usePerVertId == 1\n"
    "flat out uint fPrimitiveId;  // the id of the primitive\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "out vec4 fColor;\n"
    "#endif\n"
    "\n"
    "vec2 ScreentoNDC(in vec2 v) {\n"
    "  return vec2(((2.0 / float(viewport.width)) * (v.x - viewport.x)) - 1, ((2.0 / float(viewport.height)) * (v.y - "
    "viewport.y)) - 1);\n"
    "}\n"
    "\n"
    "void emitVertex(in vec2 pos, in vec2 zw) {\n"
    "  gl_Position = vec4(ScreentoNDC(pos), zw);\n"
    "  EmitVertex();\n"
    "}\n"
    "\n"
    "void emitBevelVertexData(\n"
    "      in vec2 pos1,\n"
    "      in vec2 zw1,\n"
    "      in vec2 miter1,\n"
    "      in vec2 halfMiter1,\n"
    "      in float halfMiterLength1,\n"
    "      in float halfThickness1,\n"
    "      in float fullThickness1,\n"
    "#if usePerVertId == 1\n"
    "      in uint id1,\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "      in vec4 color1,\n"
    "#endif\n"
    "      in vec2 pos2,\n"
    "      in vec2 zw2,\n"
    "      in vec2 miter2,\n"
    "      in vec2 halfMiter2,\n"
    "      in float halfMiterLength2,\n"
    "      in float halfThickness2,\n"
    "      in float fullThickness2,\n"
    "#if usePerVertId == 1\n"
    "      in uint id2,\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "      in vec4 color2,\n"
    "#endif\n"
    "      in vec2 lineNorm,\n"
    "      bool swap)\n"
    "{\n"
    "  vec2 halfPrevLineNorm = halfThickness1 * lineNorm;\n"
    "  vec2 fullPrevLineNorm = fullThickness1 * lineNorm;\n"
    "\n"
    "#if usePerVertId == 1\n"
    "  fId = id1;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "  fColor = color1;\n"
    "#endif\n"
    "\n"
    "  fNormDistCoords = vec2(0, 0);\n"
    "  fIsCorner = 1;\n"
    "  fNormDistCoords[0] = halfMiterLength1 / dot(halfPrevLineNorm, miter1);\n"
    "  emitVertex(pos1 + halfMiter1, zw1);\n"
    "\n"
    "  fNormDistCoords[0] = 1.0;\n"
    "  emitVertex(pos1 + halfPrevLineNorm, zw1);\n"
    "\n"
    "  // will be 0 the rest of the way out\n"
    "  fNormDistCoords[0] = 0.0;\n"
    "  emitVertex(pos1, zw1);\n"
    "\n"
    "  // will not be a vertex of a corner triangle until we write out\n"
    "  // the last tri\n"
    "  fIsCorner = 0;\n"
    "  emitVertex(pos1 - halfMiter1 + fullPrevLineNorm, zw1);\n"
    "\n"
    "  emitVertex(pos1 - halfMiter1, zw1);\n"
    "\n"
    "  vec2 halfNextLineNorm = halfThickness2 * lineNorm;\n"
    "  vec2 fullNextLineNorm = fullThickness2 * lineNorm;\n"
    "\n"
    "#if usePerVertId == 1\n"
    "  fId = id2;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "  fColor = color2;\n"
    "#endif\n"
    "\n"
    "  if (swap) {\n"
    "    emitVertex(pos2 - halfMiter2, zw2);\n"
    "    emitVertex(pos2 - halfMiter2 - fullNextLineNorm, zw2);\n"
    "\n"
    "    // will be a vertex of a corner triangle the rest of the way out\n"
    "    fIsCorner = 1;\n"
    "    emitVertex(pos2, zw2);\n"
    "\n"
    "    fNormDistCoords[0] = 1.0;\n"
    "    emitVertex(pos2 - halfNextLineNorm, zw2);\n"
    "\n"
    "    // do this if rounded corners\n"
    "    fNormDistCoords[0] = halfMiterLength2 / -dot(halfNextLineNorm, miter2);\n"
    "    emitVertex(pos2 + halfMiter2, zw2);\n"
    "  } else {\n"
    "    emitVertex(pos2 - halfMiter2 + fullNextLineNorm, zw2);\n"
    "    emitVertex(pos2 - halfMiter2, zw2);\n"
    "\n"
    "    // will be a vertex of a corner triangle the rest of the way out\n"
    "    fIsCorner = 1;\n"
    "    fNormDistCoords[0] = 1.0;\n"
    "    emitVertex(pos2 + halfNextLineNorm, zw2);\n"
    "\n"
    "    fNormDistCoords[0] = 0.0;\n"
    "    emitVertex(pos2, zw2);\n"
    "\n"
    "    fNormDistCoords[0] = halfMiterLength2 / dot(halfNextLineNorm, miter2);\n"
    "    emitVertex(pos2 + halfMiter2, zw2);\n"
    "  }\n"
    "}\n"
    "\n"
    "void emitRoundVertexData(\n"
    "      in vec2 pos1,\n"
    "      in vec2 zw1,\n"
    "      in vec2 miter1,\n"
    "      in vec2 halfMiter1,\n"
    "      in float halfMiterLength1,\n"
    "      in float halfThickness1,\n"
    "      in float fullThickness1,\n"
    "#if usePerVertId == 1\n"
    "      in uint id1,\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "      in vec4 color1,\n"
    "#endif\n"
    "      in vec2 pos2,\n"
    "      in vec2 zw2,\n"
    "      in vec2 miter2,\n"
    "      in vec2 halfMiter2,\n"
    "      in float halfMiterLength2,\n"
    "      in float halfThickness2,\n"
    "      in float fullThickness2,\n"
    "#if usePerVertId == 1\n"
    "      in uint id2,\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "      in vec4 color2,\n"
    "#endif\n"
    "      in vec2 lineNorm,\n"
    "      bool swap)\n"
    "{\n"
    "  vec2 halfPrevLineNorm = halfThickness1 * lineNorm;\n"
    "  vec2 fullPrevLineNorm = fullThickness1 * lineNorm;\n"
    "\n"
    "#if usePerVertId == 1\n"
    "  fId = id1;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "  fColor = color1;\n"
    "#endif\n"
    "\n"
    "  fIsCorner = 1;\n"
    "  fNormDistCoords = halfMiter1 / halfThickness1;\n"
    "  emitVertex(pos1 + halfMiter1, zw1);\n"
    "\n"
    "  fNormDistCoords = lineNorm;\n"
    "  emitVertex(pos1 + halfPrevLineNorm, zw1);\n"
    "\n"
    "  // will be 0 the rest of the way out\n"
    "  fNormDistCoords = vec2(0, 0);\n"
    "  emitVertex(pos1, zw1);\n"
    "\n"
    "  // will not be a vertex of a corner triangle until we write out\n"
    "  // the last tri\n"
    "  fIsCorner = 0;\n"
    "  emitVertex(pos1 - halfMiter1 + fullPrevLineNorm, zw1);\n"
    "  emitVertex(pos1 - halfMiter1, zw1);\n"
    "\n"
    "\n"
    "#if usePerVertId == 1\n"
    "  fId = id2;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "  fColor = color2;\n"
    "#endif\n"
    "\n"
    "  vec2 halfNextLineNorm = halfThickness2 * lineNorm;\n"
    "  vec2 fullNextLineNorm = fullThickness2 * lineNorm;\n"
    "\n"
    "  if (swap) {\n"
    "    emitVertex(pos2 - halfMiter2, zw2);\n"
    "    emitVertex(pos2 - halfMiter2 - fullNextLineNorm, zw2);\n"
    "\n"
    "    // will be a vertex of a corner triangle the rest of the way out\n"
    "    fIsCorner = 1;\n"
    "    emitVertex(pos2, zw2);\n"
    "\n"
    "    fNormDistCoords = -1 * lineNorm;\n"
    "    emitVertex(pos2 - halfNextLineNorm, zw2);\n"
    "\n"
    "    // do this if rounded corners\n"
    "    fNormDistCoords = halfMiter2 / halfThickness2;\n"
    "    emitVertex(pos2 + halfMiter2, zw2);\n"
    "  } else {\n"
    "    emitVertex(pos2 - halfMiter2 + fullNextLineNorm, zw2);\n"
    "    emitVertex(pos2 - halfMiter2, zw2);\n"
    "\n"
    "    // will be a vertex of a corner triangle the rest of the way out\n"
    "    fIsCorner = 1;\n"
    "    fNormDistCoords = lineNorm;\n"
    "    emitVertex(pos2 + halfNextLineNorm, zw2);\n"
    "\n"
    "    fNormDistCoords = vec2(0, 0);\n"
    "    emitVertex(pos2, zw2);\n"
    "\n"
    "    fNormDistCoords = halfMiter2 / halfThickness2;\n"
    "    emitVertex(pos2 + halfMiter2, zw2);\n"
    "  }\n"
    "}\n"
    "\n"
    "void emitMiterVertexData(\n"
    "      in vec2 pos1,\n"
    "      in vec2 zw1,\n"
    "      in vec2 miter1,\n"
    "      in vec2 halfMiter1,\n"
    "      in float halfMiterLength1,\n"
    "      in float halfThickness1,\n"
    "      in float fullThickness1,\n"
    "#if usePerVertId == 1\n"
    "      in uint id1,\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "      in vec4 color1,\n"
    "#endif\n"
    "      in vec2 pos2,\n"
    "      in vec2 zw2,\n"
    "      in vec2 miter2,\n"
    "      in vec2 halfMiter2,\n"
    "      in float halfMiterLength2,\n"
    "      in float halfThickness2,\n"
    "      in float fullThickness2,\n"
    "#if usePerVertId == 1\n"
    "      in uint id2,\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "      in vec4 color2,\n"
    "#endif\n"
    "      in vec2 lineNorm,\n"
    "      bool swap)\n"
    "{\n"
    "\n"
    "  vec2 halfPrevLineNorm = halfThickness1 * lineNorm;\n"
    "  vec2 fullPrevLineNorm = fullThickness1 * lineNorm;\n"
    "\n"
    "#if usePerVertId == 1\n"
    "  fId = id1;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "  fColor = color1;\n"
    "#endif\n"
    "\n"
    "  fNormDistCoords = vec2(0, 0);\n"
    "  fIsCorner = 1;\n"
    "  if (miterLimit > 0 && halfMiterLength1 / halfThickness1 > miterLimit) {\n"
    "    fNormDistCoords[0] = halfMiterLength1 / dot(halfPrevLineNorm, miter1);\n"
    "  } else {\n"
    "    fNormDistCoords[0] = 1.0;\n"
    "  }\n"
    "  emitVertex(pos1 + halfMiter1, zw1);\n"
    "\n"
    "  fNormDistCoords[0] = 1.0;\n"
    "  emitVertex(pos1 + halfPrevLineNorm, zw1);\n"
    "\n"
    "  // will be 0 the rest of the way out\n"
    "  fNormDistCoords[0] = 0.0;\n"
    "  emitVertex(pos1, zw1);\n"
    "\n"
    "  // will not be a vertex of a corner triangle until we write out\n"
    "  // the last tri\n"
    "  fIsCorner = 0;\n"
    "  emitVertex(pos1 - halfMiter1 + fullPrevLineNorm, zw1);\n"
    "  emitVertex(pos1 - halfMiter1, zw1);\n"
    "\n"
    "#if usePerVertId == 1\n"
    "  fId = id2;\n"
    "#endif\n"
    "\n"
    "#if usePerVertColor == 1\n"
    "  fColor = color2;\n"
    "#endif\n"
    "\n"
    "  vec2 halfNextLineNorm = halfThickness2 * lineNorm;\n"
    "  vec2 fullNextLineNorm = fullThickness2 * lineNorm;\n"
    "\n"
    "  if (swap) {\n"
    "    emitVertex(pos2 - halfMiter2, zw2);\n"
    "    emitVertex(pos2 - halfMiter2 - fullNextLineNorm, zw2);\n"
    "\n"
    "    // will be a vertex of a corner triangle the rest of the way out\n"
    "    fIsCorner = 1;\n"
    "    emitVertex(pos2, zw2);\n"
    "\n"
    "    fNormDistCoords[0] = 1.0;\n"
    "    emitVertex(pos2 - halfNextLineNorm, zw2);\n"
    "\n"
    "    if (miterLimit > 0 && halfMiterLength2 / halfThickness2 > miterLimit) {\n"
    "      fNormDistCoords[0] = halfMiterLength2 / -dot(halfNextLineNorm, miter2);\n"
    "    } else {\n"
    "      fNormDistCoords[0] = 1.0;\n"
    "    }\n"
    "    emitVertex(pos2 + halfMiter2, zw2);\n"
    "  } else {\n"
    "    emitVertex(pos2 - halfMiter2 + fullNextLineNorm, zw2);\n"
    "    emitVertex(pos2 - halfMiter2, zw2);\n"
    "\n"
    "    // will be a vertex of a corner triangle the rest of the way out\n"
    "    fIsCorner = 1;\n"
    "    fNormDistCoords[0] = 1.0;\n"
    "    emitVertex(pos2 + halfNextLineNorm, zw2);\n"
    "\n"
    "    fNormDistCoords[0] = 0.0;\n"
    "    emitVertex(pos2, zw2);\n"
    "\n"
    "    if (miterLimit > 0 && halfMiterLength2 / halfThickness2 > miterLimit) {\n"
    "      fNormDistCoords[0] = halfMiterLength2 / dot(halfNextLineNorm, miter2);\n"
    "    } else {\n"
    "      fNormDistCoords[0] = 1.0;\n"
    "    }\n"
    "    emitVertex(pos2 + halfMiter2, zw2);\n"
    "  }\n"
    "}\n"
    "\n"
    "\n"
    "\n"
    "void main()\n"
    "{\n"
    "\n"
    "  vec2 pos0 = gl_in[0].gl_Position.xy;\n"
    "  vec2 pos1 = gl_in[1].gl_Position.xy;\n"
    "  vec2 pos2 = gl_in[2].gl_Position.xy;\n"
    "  vec2 pos3 = gl_in[3].gl_Position.xy;\n"
    "\n"
    "  vec2 lineDir = pos2 - pos1;\n"
    "  if (length(lineDir) > 0) {\n"
    "\n"
    "#if usePerVertWidth == 1\n"
    "    float prevThickness = gWidth[1];\n"
    "    float nextThickness = gWidth[2];\n"
    "#elif useUstrokeWidth == 1\n"
    "    float prevThickness = float(getstrokeWidth(strokeWidth));\n"
    "    float nextThickness = prevThickness;\n"
    "#else // should be in uniform buffer\n"
    "    float prevThickness = float(getstrokeWidth(lineData.<strokeWidth>));\n"
    "    float nextThickness = prevThickness;\n"
    "#endif\n"
    "\n"
    "    float prevHalfThickness = 0.5 * prevThickness;\n"
    "    float nextHalfThickness = 0.5 * nextThickness;\n"
    "\n"
    "    lineDir = normalize(lineDir);\n"
    "\n"
    "    vec2 prevTangent = pos1 - pos0;\n"
    "    if (length(prevTangent) > 0) {\n"
    "      prevTangent = normalize(lineDir + normalize(prevTangent));\n"
    "    } else {\n"
    "      prevTangent = lineDir;\n"
    "    }\n"
    "\n"
    "    vec2 prevMiter = vec2(prevTangent.y, -prevTangent.x);\n"
    "    vec2 prevLineNorm = vec2(lineDir.y, -lineDir.x);\n"
    "    float prevDot = dot(lineDir, prevMiter);\n"
    "    if (prevDot > 0) {\n"
    "      prevMiter *= -1;\n"
    "      prevLineNorm *= -1;\n"
    "    }\n"
    "    float prevMiterHalfLength = prevHalfThickness / dot(prevMiter, prevLineNorm);\n"
    "    vec2 prevMiterHalf = prevMiterHalfLength * prevMiter;\n"
    "\n"
    "    vec2 nextTangent = pos3 - pos2;\n"
    "    if (length(nextTangent) > 0) {\n"
    "      nextTangent = normalize(normalize(nextTangent) + lineDir);\n"
    "    } else {\n"
    "      nextTangent = lineDir;\n"
    "    }\n"
    "\n"
    "    vec2 nextMiter = vec2(nextTangent.y, -nextTangent.x);\n"
    "    vec2 nextLineNorm = vec2(lineDir.y, -lineDir.x);\n"
    "    float nextDot = dot(lineDir, nextMiter);\n"
    "    if (nextDot < 0) {\n"
    "      nextMiter *= -1;\n"
    "      nextLineNorm *= -1;\n"
    "    }\n"
    "    float nextMiterHalfLength = nextHalfThickness / dot(nextMiter, nextLineNorm);\n"
    "    vec2 nextMiterHalf = nextMiterHalfLength * nextMiter;\n"
    "\n"
    "    bool swap = (dot(prevLineNorm, nextLineNorm) < 0);\n"
    "\n"
    "\n"
    "#if useUlineJoin == 1\n"
    "    if (lineJoin == 0) {\n"
    "#else // should be in uniform buffer\n"
    "    if (lineData.<lineJoin> == 0) {\n"
    "#endif\n"
    "      emitBevelVertexData(\n"
    "        pos1, gl_in[1].gl_Position.zw,\n"
    "        prevMiter, prevMiterHalf, prevMiterHalfLength,\n"
    "        prevHalfThickness, prevThickness,\n"
    "#if usePerVertId == 1\n"
    "        gId[1],\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "        gColor[1],\n"
    "#endif\n"
    "        pos2, gl_in[2].gl_Position.zw,\n"
    "        nextMiter, nextMiterHalf, nextMiterHalfLength,\n"
    "        nextHalfThickness, nextThickness,\n"
    "#if usePerVertId == 1\n"
    "        gId[2],\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "        gColor[2],\n"
    "#endif\n"
    "        prevLineNorm, swap);\n"
    "#if useUlineJoin == 1\n"
    "    } else if (lineJoin == 1) {\n"
    "#else // should be in uniform buffer\n"
    "    } else if (lineData.<lineJoin> == 1) {\n"
    "#endif\n"
    "      emitRoundVertexData(\n"
    "        pos1, gl_in[1].gl_Position.zw,\n"
    "        prevMiter, prevMiterHalf, prevMiterHalfLength,\n"
    "        prevHalfThickness, prevThickness,\n"
    "#if usePerVertId == 1\n"
    "        gId[1],\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "        gColor[1],\n"
    "#endif\n"
    "        pos2, gl_in[2].gl_Position.zw,\n"
    "        nextMiter, nextMiterHalf, nextMiterHalfLength,\n"
    "        nextHalfThickness, nextThickness,\n"
    "#if usePerVertId == 1\n"
    "        gId[2],\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "        gColor[2],\n"
    "#endif\n"
    "        prevLineNorm, swap);\n"
    "    } else {\n"
    "      emitMiterVertexData(\n"
    "        pos1, gl_in[1].gl_Position.zw,\n"
    "        prevMiter, prevMiterHalf, prevMiterHalfLength,\n"
    "        prevHalfThickness, prevThickness,\n"
    "#if usePerVertId == 1\n"
    "        gId[1],\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "        gColor[1],\n"
    "#endif\n"
    "        pos2, gl_in[2].gl_Position.zw,\n"
    "        nextMiter, nextMiterHalf, nextMiterHalfLength,\n"
    "        nextHalfThickness, nextThickness,\n"
    "#if usePerVertId == 1\n"
    "        gId[2],\n"
    "#endif\n"
    "#if usePerVertColor == 1\n"
    "        gColor[2],\n"
    "#endif\n"
    "        prevLineNorm, swap);\n"
    "    }\n"
    "\n"
    "    EndPrimitive();\n"
    "  }\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // LINETEMPLATE_GEOM_H_
