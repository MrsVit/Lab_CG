#version 450

// Входные данные вершины (из вершинного буфера)
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

// Данные объекта - дескрипторы
layout(set = 0, binding = 0) uniform Object {
    mat4 mvp;   // proj * view * model, посчитана на процессоре
    vec4 tint;  // цвет из интерфейса, умножается на цвет вершины
} obj;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = obj.mvp * vec4(inPos, 1.0);
    fragColor = inColor * obj.tint.rgb;
}
