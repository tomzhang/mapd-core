#version 450 core

layout(location = 0) out vec4 finalColor;
layout(location = 1) out uint finalId;

#define usePerVertId <usePerVertId>
#define useUid <useUid>
#if usePerVertId == 1
flat in uint fId;
#elif useUid == 1
uniform uint id;
#endif

#define usePerVertColor <usePerVertColor>
#define useUstrokeColor <useUstrokeColor>
#define inTstrokeColor <inTstrokeColorType>
#define outTstrokeColor <outTstrokeColorType>
#if usePerVertColor == 1
in vec4 fColor;
#elif useUstrokeColor == 1
uniform inTstrokeColor strokeColor;
#endif

#define useUlineJoin <useUlineJoin>
#if useUlineJoin == 1
uniform int lineJoin;
#endif

#define useUniformBuffer <useUniformBuffer>
#if useUniformBuffer == 1
<lineData>
#endif

#if usePerVertColor == 0
outTstrokeColor getstrokeColor(in inTstrokeColor strokeColor) {
  return strokeColor;
}
#endif

in vec2 fNormDistCoords;
flat in int fIsCorner;

void main() {
  if (fIsCorner == 1) {
#if useUlineJoin == 1
    if ((lineJoin == 1 && length(fNormDistCoords) > 1) ||
        (lineJoin != 1 && fNormDistCoords[0] > 1)) {
#else // should be in uniform buffer
    if ((lineData.<lineJoin> == 1 && length(fNormDistCoords) > 1) ||
        (lineData.<lineJoin> != 1 && fNormDistCoords[0] > 1)) {
#endif
      discard;
    }
  }

#if usePerVertId == 1
  finalId = fId + 1;
#elif useUid == 1
  finalId = id + 1;
#elif useUniformBuffer == 1
  finalId = lineData.<id> + 1;
#endif

#if usePerVertColor == 1
  finalColor = fColor;
#elif useUstrokeColor == 1
  finalColor = getstrokeColor(strokeColor);
#elif useUniformBuffer == 1
  finalColor = getstrokeColor(lineData.<strokeColor>);
#endif

}
