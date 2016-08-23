/**
 * Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
 * Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
 * Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
 * Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
 * Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software. As clarification, there
 * is no requirement that the copyright notice and permission be included in
 * binary distributions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef SMAANEIGHBORHOODBLENDING_FRAG_H_
#define SMAANEIGHBORHOODBLENDING_FRAG_H_

#include <string>

namespace QueryRenderer {
struct SMAANeighborhoodBlending_frag {
  static const std::string source;
};

const std::string SMAANeighborhoodBlending_frag::source =
    "#version 450 core\n"
    "\n"
    "//-----------------------------------------------------------------------------\n"
    "// Texture Access Defines\n"
    "\n"
    "#ifndef SMAA_DECODE_VELOCITY\n"
    "#define SMAA_DECODE_VELOCITY(sample) sample.rg\n"
    "#endif\n"
    "\n"
    "//-----------------------------------------------------------------------------\n"
    "// Global Uniforms\n"
    "\n"
    "uniform vec4 SMAA_RT_METRICS;\n"
    "uniform vec4 FULL_SMAA_RT_METRICS;\n"
    "\n"
    "/**\n"
    " * Conditional move:\n"
    " */\n"
    "void SMAAMovc(bvec2 cond, inout vec2 variable, vec2 value) {\n"
    "    if (cond.x) variable.x = value.x;\n"
    "    if (cond.y) variable.y = value.y;\n"
    "}\n"
    "\n"
    "void SMAAMovc(bvec4 cond, inout vec4 variable, vec4 value) {\n"
    "    SMAAMovc(cond.xy, variable.xy, value.xy);\n"
    "    SMAAMovc(cond.zw, variable.zw, value.zw);\n"
    "}\n"
    "\n"
    "//-----------------------------------------------------------------------------\n"
    "// Neighborhood Blending Pixel Shader (Third Pass)\n"
    "\n"
    "vec4 SMAANeighborhoodBlending(vec2 texcoord,\n"
    "                                vec4 offset,\n"
    "                                sampler2D colorTex,\n"
    "                                sampler2D blendTex\n"
    "                                #ifdef SMAA_REPROJECTION\n"
    "                                , sampler2D velocityTex\n"
    "                                #endif\n"
    "                                ) {\n"
    "    // Fetch the blending weights for current pixel:\n"
    "    vec4 a;\n"
    "    a.x = texture(blendTex, offset.xy).a; // Right\n"
    "    a.y = texture(blendTex, offset.zw).g; // Top\n"
    "    a.wz = texture(blendTex, texcoord).rb; // Bottom / Left\n"
    "\n"
    "    // Is there any blending weight with a value greater than 0.0?\n"
    "\n"
    "    if (dot(a, vec4(1.0, 1.0, 1.0, 1.0)) < 1e-5) {\n"
    "        vec4 color = textureLod(colorTex, texcoord, 0.0);\n"
    "\n"
    "        #ifdef SMAA_REPROJECTION\n"
    "        vec2 velocity = SMAA_DECODE_VELOCITY(textureLod(velocityTex, texcoord, 0.0));\n"
    "\n"
    "        // Pack velocity into the alpha channel:\n"
    "        color.a = sqrt(5.0 * length(velocity));\n"
    "        #endif\n"
    "\n"
    "        return color;\n"
    "    } else {\n"
    "        bool h = max(a.x, a.z) > max(a.y, a.w); // max(horizontal) > max(vertical)\n"
    "\n"
    "        // Calculate the blending offsets:\n"
    "        vec4 blendingOffset = vec4(0.0, -a.y, 0.0, -a.w);\n"
    "        vec2 blendingWeight = a.yw;\n"
    "        SMAAMovc(bvec4(h, h, h, h), blendingOffset, vec4(a.x, 0.0, a.z, 0.0));\n"
    "        SMAAMovc(bvec2(h, h), blendingWeight, a.xz);\n"
    "        blendingWeight /= dot(blendingWeight, vec2(1.0, 1.0));\n"
    "\n"
    "        // Calculate the texture coordinates:\n"
    "        vec4 blendingCoord = fma(blendingOffset, vec4(FULL_SMAA_RT_METRICS.xy, -FULL_SMAA_RT_METRICS.xy), "
    "texcoord.xyxy);\n"
    "\n"
    "        // We exploit bilinear filtering to mix current pixel with the chosen\n"
    "        // neighbor:\n"
    "        vec4 color = blendingWeight.x * textureLod(colorTex, blendingCoord.xy, 0.0);\n"
    "        color += blendingWeight.y * textureLod(colorTex, blendingCoord.zw, 0.0);\n"
    "\n"
    "        #ifdef SMAA_REPROJECTION\n"
    "        // Antialias velocity for proper reprojection in a later stage:\n"
    "        vec2 velocity = blendingWeight.x * SMAA_DECODE_VELOCITY(textureLod(velocityTex, "
    "blendingCoord.xy, 0.0));\n"
    "        velocity += blendingWeight.y * SMAA_DECODE_VELOCITY(textureLod(velocityTex, blendingCoord.zw, 0.0));\n"
    "\n"
    "        // Pack velocity into the alpha channel:\n"
    "        color.a = sqrt(5.0 * length(velocity));\n"
    "        #endif\n"
    "\n"
    "        return color;\n"
    "    }\n"
    "}\n"
    "\n"
    "/**\n"
    " * Neighborhood Blending\n"
    " */\n"
    "void SMAANeighborhoodBlendingOffset(vec2 texcoord,\n"
    "                                    out vec4 offset) {\n"
    "    offset = fma(FULL_SMAA_RT_METRICS.xyxy, vec4( 1.0, 0.0, 0.0, -1.0), texcoord.xyxy);\n"
    "}\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "\n"
    "uniform sampler2D colorTex;\n"
    "uniform sampler2D blendTex;\n"
    "\n"
    "#ifdef SMAA_REPROJECTION\n"
    "uniform sampler2D velocityTex;\n"
    "#endif\n"
    "\n"
    "in vec2 fTexCoord;\n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    vec4 offset = vec4(0.0, 0.0, 0.0, 0.0);\n"
    "    SMAANeighborhoodBlendingOffset(fTexCoord, offset);\n"
    "\n"
    "    color = SMAANeighborhoodBlending(fTexCoord,\n"
    "                                     offset,\n"
    "                                     colorTex,\n"
    "                                     blendTex\n"
    "                                     #ifdef SMAA_REPROJECTION\n"
    "                                     , velocityTex\n"
    "                                     #endif\n"
    "                                     );\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // SMAANEIGHBORHOODBLENDING_FRAG_H_
