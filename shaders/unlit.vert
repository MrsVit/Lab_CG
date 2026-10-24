#version 450

// Из вершины нужна только позиция
layout(location = 0) in vec3 inPos;

// Данные приходят через push constants: матрица и цвет фигуры
layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 color;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}
