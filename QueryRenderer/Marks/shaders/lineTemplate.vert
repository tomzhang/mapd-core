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
#define useUstrokeColor <useUstrokeColor>
#define inTstrokeColor <inTstrokeColorType>
#define outTstrokeColor <outTstrokeColorType>

#if useUstrokeColor == 1
uniform inTstrokeColor strokeColor;
#elif usePerVertColor == 1
in inTstrokeColor strokeColor;
#endif

#define usePerVertWidth <usePerVertWidth>
#define useUstrokeWidth <useUstrokeWidth>
#define inTstrokeWidth <inTstrokeWidthType>
#define outTstrokeWidth <outTstrokeWidthType>

#if useUstrokeWidth == 1
uniform inTstrokeWidth strokeWidth;
#elif usePerVertWidth == 1
in inTstrokeWidth strokeWidth;
#endif

#define useUniformBuffer <useUniformBuffer>
#if useUniformBuffer == 1
<lineData>
#endif

outTx getx(in inTx x) {
  return x;
}

outTy gety(in inTy y) {
  return y;
}

outTstrokeColor getstrokeColor(in inTstrokeColor strokeColor) {
  return strokeColor;
}

outTstrokeWidth getstrokeWidth(in inTstrokeWidth strokeWidth)  {
  return strokeWidth;
}

////////////////////////////////////////////////////////////////
/**
 * Non-interpolated shader outputs.
 */
flat out uint fPrimitiveId;  // the id of the primitive

#if usePerVertColor == 1
out vec4 fColor;
#else
flat out vec4 fColor;
#endif

void main() {
  gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);

#if usePerVertColor == 1 || useUstrokeColor
  fColor = getstrokeColor(strokeColor);
#elif useUniformBuffer == 1
  fColor = getstrokeColor(lineData.strokeColor);
#endif

  // ids from queries go from 0 to numrows-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
#if useUid == 1
  fPrimitiveId = id + 1;
#elif useUniformBuffer == 1
  fPrimitiveId = lineData.id + 1;
#endif
}
