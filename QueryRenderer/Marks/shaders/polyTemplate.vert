// VERTEX SHADER

#version 410 core

#define inTx <inTxType>
#define outTx <outTxType>

#define inTy <inTyType>
#define outTy <outTyType>

#define useUid <useUid>

in inTx x;
in inTy y;

#if useUid == 1
uniform uint id;
#endif

#define usePerVertColor <usePerVertColor>
#define useUfillColor <useUfillColor>
#define inTfillColor <inTfillColorType>
#define outTfillColor <outTfillColorType>

#if useUfillColor == 1
uniform inTfillColor fillColor;
#elif usePerVertColor == 1
in inTfillColor fillColor;
#endif

#define useUniformBuffer <useUniformBuffer>
#if useUniformBuffer == 1
layout(std140) uniform PolyData {
#if !usePerVertColor && !useUfillColor
  inTfillColor fillColor;
#endif
#if !useUid
  uint id;
#endif
} polyData;
#endif


outTx getx(in inTx x) {
  return x;
}

outTy gety(in inTy y) {
  return y;
}

outTfillColor getfillColor(in inTfillColor fillColor) {
  return fillColor;
}

////////////////////////////////////////////////////////////////
/**
 * Non-interpolated shader outputs.
 */
flat out uint fPrimitiveId;  // the id of the primitive

// the output color of the primitive
#if usePerVertColor == 1
out vec4 fColor;
#else
flat out vec4 fColor;
#endif

void main() {
  gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);

  #if usePerVertColor == 1 || useUfillColor
  fColor = getfillColor(fillColor);
  #elif useUniformBuffer == 1
  fColor = getfillColor(polyData.fillColor);
  #endif

  // ids from queries go from 0 to numrows-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
  #if useUid == 1
  fPrimitiveId = id + 1;
  #elif useUniformBuffer == 1
  fPrimitiveId = polyData.id + 1;
  #endif
}
