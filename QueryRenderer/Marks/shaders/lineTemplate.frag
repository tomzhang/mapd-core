#version 450 core

layout(location = 0) out vec4 color;
layout(location = 1) out uint id;

flat in uint fPrimitiveId;

#define usePerVertColor <usePerVertColor>
#if usePerVertColor == 1
in vec4 fColor;
#else
flat in vec4 fColor;
#endif

void main() {
    // TODO(croot): do some kind of temporary anti-aliasing?
    color = fColor;
    id = fPrimitiveId;
}
