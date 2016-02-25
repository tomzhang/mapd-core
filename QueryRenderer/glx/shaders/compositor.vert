// VERTEX SHADER

#version 450 core

in vec2 pos;
in vec2 texCoords;

out vec2 oTexCoords;        // the output color of the primitive

void main() {
    gl_Position = vec4(pos.x, pos.y, 0.5, 1.0);
    oTexCoords = texCoords;
}
