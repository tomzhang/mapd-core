// VERTEX SHADER

#version 450 core

in vec2 pos;

void main() {
    gl_Position = vec4(pos.x, pos.y, 0.5, 1.0);
}
