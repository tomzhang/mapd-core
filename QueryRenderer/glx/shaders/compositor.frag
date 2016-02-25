#version 450 core

layout(location = 0) out vec4 color;

uniform sampler2DArray texArraySampler;
uniform int texArraySize; // can extract the size of the sampler using the textureSize(...) call, but I'm thinking this could be slightly faster

in vec2 oTexCoords;

void main() {
    vec4 dstColor = vec4(0,0,0,0);
    vec4 srcColor = vec4(0,0,0,0);
    float srcFactorRGB;
    float dstFactorRGB;
    float srcFactorA = 1.0;
    float dstFactorA;


    int i = 0;
    while (i < texArraySize) {

      // doing our own blend here to avoid mulitple texture lookups.
      // We're doing the same blend according to:
      // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
      // glBlendEquationsSeparate(GL_ADD, GL_ADD);
      // This is what is beind done in the query rendering, but if that
      // were to ever change, we'd have to change that here as well.
      // We should look at ways of exposing all those options as subroutines.

      srcColor = texture(texArraySampler, vec3(oTexCoords, i));

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

    color = dstColor;
}
