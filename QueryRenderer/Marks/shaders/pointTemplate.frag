#version 450 core

layout(location = 0) out vec4 color;
layout(location = 1) out uint id;

flat in uint fPrimitiveId;
flat in vec4 fColor;
flat in float fPointSize;

void main() {
    float dist = distance(gl_PointCoord, vec2(0.5));

    if (fPointSize < 3.0) {
        if (dist > 0.5) {
            discard;
        }
        color = fColor;
    } else {
        float delta = fwidth(dist);
        float alpha = 1.0 - smoothstep(0.5-delta, 0.5, dist);
        if (alpha == 0.0) {
            discard;
        }

        color = vec4(fColor.xyz, fColor.w*alpha);
    }

    id = fPrimitiveId;
}

