// GEOMETRY SHADER

#version 450 core

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 10) out;

struct Viewport
{
  int x;
  int y;
  int width;
  int height;
};
uniform Viewport viewport;

#define usePerVertId <usePerVertId>
#define useUid <useUid>
#if usePerVertId == 1
in uint gId[];
#elif useUid == 1
uniform uint id;
#endif

#define usePerVertColor <usePerVertColor>
#if usePerVertColor == 1
in vec4 gColor[];
#endif

#define usePerVertWidth <usePerVertWidth>
#define useUstrokeWidth <useUstrokeWidth>
#define inTstrokeWidth <inTstrokeWidthType>
#define outTstrokeWidth <outTstrokeWidthType>
#if usePerVertWidth == 1
in float gWidth[];
#elif useUstrokeWidth == 1
uniform inTstrokeWidth strokeWidth;
#endif

#define useUlineJoin <useUlineJoin>
#if useUlineJoin == 1
uniform int lineJoin;
#endif

#define inTmiterLimit <inTmiterLimitType>
#define outTmiterLimit <outTmiterLimitType>
#define useUmiterLimit <useUmiterLimit>
#if useUmiterLimit == 1
uniform inTmiterLimit miterLimit;
#endif

#define useUniformBuffer <useUniformBuffer>
#if useUniformBuffer == 1
<lineData>
#endif

#if usePerVertWidth == 0
outTstrokeWidth getstrokeWidth(in inTstrokeWidth strokeWidth)  {
  return strokeWidth;
}
#endif

outTmiterLimit getmiterLimit(in inTmiterLimit miterLimit) {
  return miterLimit;
}

out vec2 fNormDistCoords;
flat out int fIsCorner;

#if usePerVertId == 1
flat out uint fPrimitiveId;  // the id of the primitive
#endif

#if usePerVertColor == 1
out vec4 fColor;
#endif

vec2 ScreentoNDC(in vec2 v) {
  return vec2(((2.0 / float(viewport.width)) * (v.x - viewport.x)) - 1, ((2.0 / float(viewport.height)) * (v.y - viewport.y)) - 1);
}

void emitVertex(in vec2 pos, in vec2 zw) {
  gl_Position = vec4(ScreentoNDC(pos), zw);
  EmitVertex();
}

void emitBevelVertexData(
      in vec2 pos1,
      in vec2 zw1,
      in vec2 miter1,
      in vec2 halfMiter1,
      in float halfMiterLength1,
      in float halfThickness1,
      in float fullThickness1,
#if usePerVertId == 1
      in uint id1,
#endif
#if usePerVertColor == 1
      in vec4 color1,
#endif
      in vec2 pos2,
      in vec2 zw2,
      in vec2 miter2,
      in vec2 halfMiter2,
      in float halfMiterLength2,
      in float halfThickness2,
      in float fullThickness2,
#if usePerVertId == 1
      in uint id2,
#endif
#if usePerVertColor == 1
      in vec4 color2,
#endif
      in vec2 lineNorm,
      bool swap)
{
  vec2 halfPrevLineNorm = halfThickness1 * lineNorm;
  vec2 fullPrevLineNorm = fullThickness1 * lineNorm;

#if usePerVertId == 1
  fId = id1;
#endif

#if usePerVertColor == 1
  fColor = color1;
#endif

  fNormDistCoords = vec2(0, 0);
  fIsCorner = 1;
  fNormDistCoords[0] = halfMiterLength1 / dot(halfPrevLineNorm, miter1);
  emitVertex(pos1 + halfMiter1, zw1);

  fNormDistCoords[0] = 1.0;
  emitVertex(pos1 + halfPrevLineNorm, zw1);

  // will be 0 the rest of the way out
  fNormDistCoords[0] = 0.0;
  emitVertex(pos1, zw1);

  // will not be a vertex of a corner triangle until we write out
  // the last tri
  fIsCorner = 0;
  emitVertex(pos1 - halfMiter1 + fullPrevLineNorm, zw1);

  emitVertex(pos1 - halfMiter1, zw1);

  vec2 halfNextLineNorm = halfThickness2 * lineNorm;
  vec2 fullNextLineNorm = fullThickness2 * lineNorm;

#if usePerVertId == 1
  fId = id2;
#endif

#if usePerVertColor == 1
  fColor = color2;
#endif

  if (swap) {
    emitVertex(pos2 - halfMiter2, zw2);
    emitVertex(pos2 - halfMiter2 - fullNextLineNorm, zw2);

    // will be a vertex of a corner triangle the rest of the way out
    fIsCorner = 1;
    emitVertex(pos2, zw2);

    fNormDistCoords[0] = 1.0;
    emitVertex(pos2 - halfNextLineNorm, zw2);

    // do this if rounded corners
    fNormDistCoords[0] = halfMiterLength2 / -dot(halfNextLineNorm, miter2);
    emitVertex(pos2 + halfMiter2, zw2);
  } else {
    emitVertex(pos2 - halfMiter2 + fullNextLineNorm, zw2);
    emitVertex(pos2 - halfMiter2, zw2);

    // will be a vertex of a corner triangle the rest of the way out
    fIsCorner = 1;
    fNormDistCoords[0] = 1.0;
    emitVertex(pos2 + halfNextLineNorm, zw2);

    fNormDistCoords[0] = 0.0;
    emitVertex(pos2, zw2);

    fNormDistCoords[0] = halfMiterLength2 / dot(halfNextLineNorm, miter2);
    emitVertex(pos2 + halfMiter2, zw2);
  }
}

void emitRoundVertexData(
      in vec2 pos1,
      in vec2 zw1,
      in vec2 miter1,
      in vec2 halfMiter1,
      in float halfMiterLength1,
      in float halfThickness1,
      in float fullThickness1,
#if usePerVertId == 1
      in uint id1,
#endif
#if usePerVertColor == 1
      in vec4 color1,
#endif
      in vec2 pos2,
      in vec2 zw2,
      in vec2 miter2,
      in vec2 halfMiter2,
      in float halfMiterLength2,
      in float halfThickness2,
      in float fullThickness2,
#if usePerVertId == 1
      in uint id2,
#endif
#if usePerVertColor == 1
      in vec4 color2,
#endif
      in vec2 lineNorm,
      bool swap)
{
  vec2 halfPrevLineNorm = halfThickness1 * lineNorm;
  vec2 fullPrevLineNorm = fullThickness1 * lineNorm;

#if usePerVertId == 1
  fId = id1;
#endif

#if usePerVertColor == 1
  fColor = color1;
#endif

  fIsCorner = 1;
  fNormDistCoords = halfMiter1 / halfThickness1;
  emitVertex(pos1 + halfMiter1, zw1);

  fNormDistCoords = lineNorm;
  emitVertex(pos1 + halfPrevLineNorm, zw1);

  // will be 0 the rest of the way out
  fNormDistCoords = vec2(0, 0);
  emitVertex(pos1, zw1);

  // will not be a vertex of a corner triangle until we write out
  // the last tri
  fIsCorner = 0;
  emitVertex(pos1 - halfMiter1 + fullPrevLineNorm, zw1);
  emitVertex(pos1 - halfMiter1, zw1);


#if usePerVertId == 1
  fId = id2;
#endif

#if usePerVertColor == 1
  fColor = color2;
#endif

  vec2 halfNextLineNorm = halfThickness2 * lineNorm;
  vec2 fullNextLineNorm = fullThickness2 * lineNorm;

  if (swap) {
    emitVertex(pos2 - halfMiter2, zw2);
    emitVertex(pos2 - halfMiter2 - fullNextLineNorm, zw2);

    // will be a vertex of a corner triangle the rest of the way out
    fIsCorner = 1;
    emitVertex(pos2, zw2);

    fNormDistCoords = -1 * lineNorm;
    emitVertex(pos2 - halfNextLineNorm, zw2);

    // do this if rounded corners
    fNormDistCoords = halfMiter2 / halfThickness2;
    emitVertex(pos2 + halfMiter2, zw2);
  } else {
    emitVertex(pos2 - halfMiter2 + fullNextLineNorm, zw2);
    emitVertex(pos2 - halfMiter2, zw2);

    // will be a vertex of a corner triangle the rest of the way out
    fIsCorner = 1;
    fNormDistCoords = lineNorm;
    emitVertex(pos2 + halfNextLineNorm, zw2);

    fNormDistCoords = vec2(0, 0);
    emitVertex(pos2, zw2);

    fNormDistCoords = halfMiter2 / halfThickness2;
    emitVertex(pos2 + halfMiter2, zw2);
  }
}

void emitMiterVertexData(
      in vec2 pos1,
      in vec2 zw1,
      in vec2 miter1,
      in vec2 halfMiter1,
      in float halfMiterLength1,
      in float halfThickness1,
      in float fullThickness1,
#if usePerVertId == 1
      in uint id1,
#endif
#if usePerVertColor == 1
      in vec4 color1,
#endif
      in vec2 pos2,
      in vec2 zw2,
      in vec2 miter2,
      in vec2 halfMiter2,
      in float halfMiterLength2,
      in float halfThickness2,
      in float fullThickness2,
#if usePerVertId == 1
      in uint id2,
#endif
#if usePerVertColor == 1
      in vec4 color2,
#endif
      in vec2 lineNorm,
      bool swap)
{

  vec2 halfPrevLineNorm = halfThickness1 * lineNorm;
  vec2 fullPrevLineNorm = fullThickness1 * lineNorm;

#if usePerVertId == 1
  fId = id1;
#endif

#if usePerVertColor == 1
  fColor = color1;
#endif

  fNormDistCoords = vec2(0, 0);
  fIsCorner = 1;
  if (miterLimit > 0 && halfMiterLength1 / halfThickness1 > miterLimit) {
    fNormDistCoords[0] = halfMiterLength1 / dot(halfPrevLineNorm, miter1);
  } else {
    fNormDistCoords[0] = 1.0;
  }
  emitVertex(pos1 + halfMiter1, zw1);

  fNormDistCoords[0] = 1.0;
  emitVertex(pos1 + halfPrevLineNorm, zw1);

  // will be 0 the rest of the way out
  fNormDistCoords[0] = 0.0;
  emitVertex(pos1, zw1);

  // will not be a vertex of a corner triangle until we write out
  // the last tri
  fIsCorner = 0;
  emitVertex(pos1 - halfMiter1 + fullPrevLineNorm, zw1);
  emitVertex(pos1 - halfMiter1, zw1);

#if usePerVertId == 1
  fId = id2;
#endif

#if usePerVertColor == 1
  fColor = color2;
#endif

  vec2 halfNextLineNorm = halfThickness2 * lineNorm;
  vec2 fullNextLineNorm = fullThickness2 * lineNorm;

  if (swap) {
    emitVertex(pos2 - halfMiter2, zw2);
    emitVertex(pos2 - halfMiter2 - fullNextLineNorm, zw2);

    // will be a vertex of a corner triangle the rest of the way out
    fIsCorner = 1;
    emitVertex(pos2, zw2);

    fNormDistCoords[0] = 1.0;
    emitVertex(pos2 - halfNextLineNorm, zw2);

    if (miterLimit > 0 && halfMiterLength2 / halfThickness2 > miterLimit) {
      fNormDistCoords[0] = halfMiterLength2 / -dot(halfNextLineNorm, miter2);
    } else {
      fNormDistCoords[0] = 1.0;
    }
    emitVertex(pos2 + halfMiter2, zw2);
  } else {
    emitVertex(pos2 - halfMiter2 + fullNextLineNorm, zw2);
    emitVertex(pos2 - halfMiter2, zw2);

    // will be a vertex of a corner triangle the rest of the way out
    fIsCorner = 1;
    fNormDistCoords[0] = 1.0;
    emitVertex(pos2 + halfNextLineNorm, zw2);

    fNormDistCoords[0] = 0.0;
    emitVertex(pos2, zw2);

    if (miterLimit > 0 && halfMiterLength2 / halfThickness2 > miterLimit) {
      fNormDistCoords[0] = halfMiterLength2 / dot(halfNextLineNorm, miter2);
    } else {
      fNormDistCoords[0] = 1.0;
    }
    emitVertex(pos2 + halfMiter2, zw2);
  }
}



void main()
{
  vec2 pos0 = gl_in[0].gl_Position.xy;
  vec2 pos1 = gl_in[1].gl_Position.xy;
  vec2 pos2 = gl_in[2].gl_Position.xy;
  vec2 pos3 = gl_in[3].gl_Position.xy;

  vec2 lineDir = pos2 - pos1;
  if (length(lineDir) > 0) {

#if usePerVertWidth == 1
    float prevThickness = gWidth[1];
    float nextThickness = gWidth[2];
#elif useUstrokeWidth == 1
    float prevThickness = float(getstrokeWidth(strokeWidth));
    float nextThickness = prevThickness;
#else // should be in uniform buffer
    float prevThickness = float(getstrokeWidth(lineData.<strokeWidth>));
    float nextThickness = prevThickness;
#endif

    float prevHalfThickness = 0.5 * prevThickness;
    float nextHalfThickness = 0.5 * nextThickness;

    lineDir = normalize(lineDir);

    vec2 prevTangent = pos1 - pos0;
    if (length(prevTangent) > 0) {
      prevTangent = normalize(lineDir + normalize(prevTangent));
    } else {
      prevTangent = lineDir;
    }

    vec2 prevMiter = vec2(prevTangent.y, -prevTangent.x);
    vec2 prevLineNorm = vec2(lineDir.y, -lineDir.x);
    float prevDot = dot(lineDir, prevMiter);
    if (prevDot > 0) {
      prevMiter *= -1;
      prevLineNorm *= -1;
    }
    float prevMiterHalfLength = prevHalfThickness / dot(prevMiter, prevLineNorm);
    vec2 prevMiterHalf = prevMiterHalfLength * prevMiter;

    vec2 nextTangent = pos3 - pos2;
    if (length(nextTangent) > 0) {
      nextTangent = normalize(normalize(nextTangent) + lineDir);
    } else {
      nextTangent = lineDir;
    }

    vec2 nextMiter = vec2(nextTangent.y, -nextTangent.x);
    vec2 nextLineNorm = vec2(lineDir.y, -lineDir.x);
    float nextDot = dot(lineDir, nextMiter);
    if (nextDot < 0) {
      nextMiter *= -1;
      nextLineNorm *= -1;
    }
    float nextMiterHalfLength = nextHalfThickness / dot(nextMiter, nextLineNorm);
    vec2 nextMiterHalf = nextMiterHalfLength * nextMiter;

    bool swap = (dot(prevLineNorm, nextLineNorm) < 0);


#if useUlineJoin == 1
    if (lineJoin == 0) {
#else // should be in uniform buffer
    if (lineData.<lineJoin> == 0) {
#endif
      emitBevelVertexData(
        pos1, gl_in[1].gl_Position.zw,
        prevMiter, prevMiterHalf, prevMiterHalfLength,
        prevHalfThickness, prevThickness,
#if usePerVertId == 1
        gId[1],
#endif
#if usePerVertColor == 1
        gColor[1],
#endif
        pos2, gl_in[2].gl_Position.zw,
        nextMiter, nextMiterHalf, nextMiterHalfLength,
        nextHalfThickness, nextThickness,
#if usePerVertId == 1
        gId[2],
#endif
#if usePerVertColor == 1
        gColor[2],
#endif
        prevLineNorm, swap);
#if useUlineJoin == 1
    } else if (lineJoin == 1) {
#else // should be in uniform buffer
    } else if (lineData.<lineJoin> == 1) {
#endif
      emitRoundVertexData(
        pos1, gl_in[1].gl_Position.zw,
        prevMiter, prevMiterHalf, prevMiterHalfLength,
        prevHalfThickness, prevThickness,
#if usePerVertId == 1
        gId[1],
#endif
#if usePerVertColor == 1
        gColor[1],
#endif
        pos2, gl_in[2].gl_Position.zw,
        nextMiter, nextMiterHalf, nextMiterHalfLength,
        nextHalfThickness, nextThickness,
#if usePerVertId == 1
        gId[2],
#endif
#if usePerVertColor == 1
        gColor[2],
#endif
        prevLineNorm, swap);
    } else {
      emitMiterVertexData(
        pos1, gl_in[1].gl_Position.zw,
        prevMiter, prevMiterHalf, prevMiterHalfLength,
        prevHalfThickness, prevThickness,
#if usePerVertId == 1
        gId[1],
#endif
#if usePerVertColor == 1
        gColor[1],
#endif
        pos2, gl_in[2].gl_Position.zw,
        nextMiter, nextMiterHalf, nextMiterHalfLength,
        nextHalfThickness, nextThickness,
#if usePerVertId == 1
        gId[2],
#endif
#if usePerVertColor == 1
        gColor[2],
#endif
        prevLineNorm, swap);
    }

    EndPrimitive();
  }
}
