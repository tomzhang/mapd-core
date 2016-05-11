// VERTEX SHADER

#version 450 core

#define inTx <inTxType>
#define outTx <outTxType>

#define inTy <inTyType>
#define outTy <outTyType>

#define usePerVertId <usePerVertId>
#if usePerVertId == 1
in uint id;
#endif


in inTx x;
in inTy y;

#define usePerVertColor <usePerVertColor>
#define inTstrokeColor <inTstrokeColorType>
#define outTstrokeColor <outTstrokeColorType>

#if usePerVertColor == 1
in inTstrokeColor strokeColor;
#endif

#define usePerVertWidth <usePerVertWidth>
#define inTstrokeWidth <inTstrokeWidthType>
#define outTstrokeWidth <outTstrokeWidthType>

#if usePerVertWidth == 1
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

#if usePerVertColor == 1
outTstrokeColor getstrokeColor(in inTstrokeColor strokeColor) {
  return strokeColor;
}
#endif

#if usePerVertWidth == 1
outTstrokeWidth getstrokeWidth(in inTstrokeWidth strokeWidth)  {
  return strokeWidth;
}
#endif

// viewport data
struct Viewport
{
    int x;
    int y;
    int width;
    int height;
};
uniform Viewport viewport;

vec2 NDCtoScreen(in outTx x, in outTy y) {
  return vec2(float(x + 1) * (float(viewport.width) / 2.0) + viewport.x, float(y + 1) * (float(viewport.height) / 2.0) + viewport.y);
}

#if usePerVertId == 1
out uint gId;
#endif

#if usePerVertColor == 1
out vec4 gColor;
#endif

#if usePerVertWidth == 1
out float gWidth;
#endif

void main() {
  gl_Position = vec4(NDCtoScreen(getx(x), gety(y)), 0.5, 1.0);

#if usePerVertId
  gId = id;
#endif

#if usePerVertColor
  gColor = getstrokeColor(strokeColor);
#endif

#if usePerVertWidth
  gWidth = getstrokeWidth(strokeWidth);
#endif
}
