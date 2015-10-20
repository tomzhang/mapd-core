#version 410 core

layout(location = 0) out vec4 color;
layout(location = 1) out uint id;

flat in uint fPrimitiveId;
flat in vec4 fColor;

void main() {
    float dist = distance(gl_PointCoord, vec2(0.5));
    if (dist > 0.5) {
        discard;
    }

    color = fColor;
    id = fPrimitiveId;
}
