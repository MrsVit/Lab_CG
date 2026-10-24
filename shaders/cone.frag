#version 450

// Входные данные из вершинного шейдера
layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

// Выходной цвет пикселя
layout(location = 0) out vec4 outColor;

// Набор дескрипторов 0 - сцена (то же описание, что и в вершинном шейдере)
layout(set = 0, binding = 0) uniform Scene {
    mat4 view_proj;
    vec4 camera_pos;
    vec4 sun_dir;
    vec4 sun_color;
    vec4 ambient;
    int point_count;
    int spot_count;
} scene;

// Набор дескрипторов 1 - модель
layout(set = 1, binding = 0) uniform Model {
    mat4 model;
    mat4 normal_matrix;
    vec4 albedo;
    vec4 specular;
} obj;

// Источники света лежат в storage буферах, размер массива в шейдере не задан:
// сколько источников использовать, говорят point_count и spot_count
struct PointLight {
    vec4 position;   // xyz - положение
    vec4 color;      // rgb - цвет, w - интенсивность
};

struct SpotLight {
    vec4 position;   // xyz - положение
    vec4 direction;  // xyz - куда светит
    vec4 color;      // rgb - цвет, w - интенсивность
    vec4 angles;     // x - косинус внутреннего угла, y - косинус внешнего угла
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight lights[];
} points;

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight lights[];
} spots;

// Модель Блинна-Фонга для одного источника.
// n - нормаль, v - к камере, l - к источнику, light - цвет источника с учётом интенсивности
vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 light, vec3 albedo) {
    float diff = max(dot(n, l), 0.0);

    float spec = 0.0;
    if (diff > 0.0) {
        vec3 h = normalize(l + v);
        spec = pow(max(dot(n, h), 0.0), obj.specular.w);
    }

    return (albedo * diff + obj.specular.rgb * spec) * light;
}

void main() {
    vec3 n = normalize(fragNormal);
    vec3 v = normalize(scene.camera_pos.xyz - fragPos);

    vec3 albedo = obj.albedo.rgb;
    if (obj.albedo.w > 0.5) {
        albedo *= fragColor;
    }

    // рассеянный свет
    vec3 result = albedo * scene.ambient.rgb * scene.ambient.w;

    // направленный источник: направление одно для всей сцены, затухания нет
    vec3 sun_l = normalize(-scene.sun_dir.xyz);
    result += blinnPhong(n, v, sun_l, scene.sun_color.rgb * scene.sun_color.w, albedo);

    // точечные источники, затухание по закону обратных квадратов
    for (int i = 0; i < scene.point_count; ++i) {
        vec3 to_light = points.lights[i].position.xyz - fragPos;
        float dist2 = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light * inversesqrt(dist2);

        vec3 light = points.lights[i].color.rgb * points.lights[i].color.w / dist2;
        result += blinnPhong(n, v, l, light, albedo);
    }

    // прожекторные источники: то же затухание и мягкий край освещённой области
    for (int i = 0; i < scene.spot_count; ++i) {
        vec3 to_light = spots.lights[i].position.xyz - fragPos;
        float dist2 = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light * inversesqrt(dist2);

        // косинус угла между осью прожектора и лучом к точке
        float cos_angle = dot(-l, normalize(spots.lights[i].direction.xyz));
        float cos_inner = spots.lights[i].angles.x;
        float cos_outer = spots.lights[i].angles.y;

        // 1 внутри внутреннего угла, 0 за внешним, между ними плавный переход
        float t = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
        float edge = t * t * (3.0 - 2.0 * t);

        vec3 light = spots.lights[i].color.rgb * spots.lights[i].color.w / dist2 * edge;
        result += blinnPhong(n, v, l, light, albedo);
    }

    outColor = vec4(result, 1.0);
}
