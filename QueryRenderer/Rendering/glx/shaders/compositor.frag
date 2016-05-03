#version 450 core

layout(location = 0) out vec4 color;
layout(location = 1) out uint id;

#define doMultiSample <doMultiSample>

#if doMultiSample==1
uniform sampler2DMSArray rgbaArraySampler;
uniform usampler2DMSArray idArraySampler;
#else
uniform sampler2DArray rgbaArraySampler;
uniform usampler2DArray idArraySampler;
#endif

uniform int rgbaArraySize;
uniform int idArraySize;

void main() {
    vec4 dstColor = vec4(0,0,0,0);
    vec4 srcColor = vec4(0,0,0,0);
    float srcFactorRGB;
    float dstFactorRGB;
    float srcFactorA = 1.0;
    float dstFactorA;

    uint primitveId = 0;
    int i = 0;

    while (i < rgbaArraySize) {

      // doing our own blend here to avoid mulitple texture lookups.
      // We're doing the same blend according to:
      // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
      // glBlendEquationsSeparate(GL_ADD, GL_ADD);
      // This is what is beind done in the query rendering, but if that
      // were to ever change, we'd have to change that here as well.
      // We should look at ways of exposing all those options as subroutines.

#if doMultiSample == 1
      srcColor = texelFetch(rgbaArraySampler, ivec3(gl_FragCoord.xy, i), gl_SampleID);
#else
      srcColor = texelFetch(rgbaArraySampler, ivec3(gl_FragCoord.xy, i), 0);
#endif

      if (srcColor.a > 0.0) {
        if (srcColor.a < 1.0) {
          if (dstColor.a > 0.0) {
            srcFactorRGB = srcColor.a;
            dstFactorRGB = 1 - srcColor.a;
            dstFactorA = dstFactorRGB;

            dstColor = vec4(srcFactorRGB * srcColor.rgb + dstFactorRGB * dstColor.rgb, srcFactorA * srcColor.a + dstFactorA * dstColor.a);

            if (dstColor.a >= 1.0) {
              break;
            }
          } else {
            dstColor = srcColor;
          }
        } else {
          dstColor = srcColor;
          break;
        }
      }

      ++i;
    }

    if (idArraySize > 0) {
#if doMultiSample == 1
      primitveId = texelFetch(idArraySampler, ivec3(gl_FragCoord.xy, i), gl_SampleID).r;
#else
      primitveId = texelFetch(idArraySampler, ivec3(gl_FragCoord.xy, i), 0).r;
#endif
    }

    color = dstColor;
    id = primitveId;
}
