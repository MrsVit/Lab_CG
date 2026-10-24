#version 450

// Входные данные вершины (из вершинного буфера)
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

// Набор дескрипторов 0 - сцена: данные, общие для всех объектов
layout(set = 0, binding = 0) uniform Scene {
    mat4 view_proj;    // proj * view, посчитана на процессоре
    vec4 camera_pos;   // положение камеры
    vec4 sun_dir;      // куда светит направленный источник
    vec4 sun_color;    // rgb - цвет, w - интенсивность
    vec4 ambient;      // rgb - цвет рассеянного света, w - интенсивность
    int point_count;   // сколько точечных источников лежит в storage буфере
    int spot_count;    // сколько прожекторных источников лежит в storage буфере
} scene;

// Набор дескрипторов 1 - модель: свой набор у каждого объекта
layout(set = 1, binding = 0) uniform Model {
    mat4 model;          // из локальных координат в мировые
    mat4 normal_matrix;  // для нормалей (поворот и обратное растяжение)
    vec4 albedo;         // rgb - альбедо, w = 1 значит "умножать на цвет вершины"
    vec4 specular;       // rgb - цвет блика, w - коэффициент блика
} obj;

// Выходные данные для фрагментного шейдера
layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main() {
    vec4 world = obj.model * vec4(inPos, 1.0);

    gl_Position = scene.view_proj * world;

    fragPos = world.xyz;
    fragNormal = mat3(obj.normal_matrix) * inNormal;
    fragColor = inColor;
}
