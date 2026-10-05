#pragma once

#include <cmath>

// Матрица 4x4 хранится по столбцам (в GLSL вроде также):
// элемент из строки r и столбца c лежит в m[c * 4 + r].
// Вершина - это столбец (x, y, z, 1), поэтому преобразования пишутся справа налево:

struct Mat4 {
	float m[16];
};

constexpr float pi = 3.14159265f;

inline float to_rad(float deg) {
	return deg * pi / 180.0f;
}

inline Mat4 mat_identity() {
	Mat4 r = {};
	r.m[0] = 1.0f;
	r.m[5] = 1.0f;
	r.m[10] = 1.0f;
	r.m[15] = 1.0f;
	return r;
}


inline Mat4 mat_mul(const Mat4& a, const Mat4& b) {
	Mat4 r = {};
	for (int c = 0; c < 4; ++c) {
		for (int row = 0; row < 4; ++row) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.m[k * 4 + row] * b.m[c * 4 + k];
			}
			r.m[c * 4 + row] = sum;
		}
	}
	return r;
}

// Сдвиг (линейный перенос) а вектор (x, y, z)
inline Mat4 mat_translate(float x, float y, float z) {
	Mat4 r = mat_identity();
	r.m[12] = x;
	r.m[13] = y;
	r.m[14] = z;
	return r;
}

// Растяжение по осb/ям
inline Mat4 mat_scale(float x, float y, float z) {
	Mat4 r = mat_identity();
	r.m[0] = x;
	r.m[5] = y;
	r.m[10] = z;
	return r;
}

// Повороты вокруг осей (угол передаём в радианах (выше есть функция перевода), правая тройка координат)
inline Mat4 mat_rotate_x(float a) {
	const float c = std::cos(a);
	const float s = std::sin(a);
	Mat4 r = mat_identity();
	r.m[5] = c;
	r.m[9] = -s;
	r.m[6] = s;
	r.m[10] = c;
	return r;
}

//тоже самое по у

inline Mat4 mat_rotate_y(float a) {
	const float c = std::cos(a);
	const float s = std::sin(a);
	Mat4 r = mat_identity();
	r.m[0] = c;
	r.m[8] = s;
	r.m[2] = -s;
	r.m[10] = c;
	return r;
}

//тоже самое но по z
inline Mat4 mat_rotate_z(float a) {
	const float c = std::cos(a);
	const float s = std::sin(a);
	Mat4 r = mat_identity();
	r.m[0] = c;
	r.m[4] = -s;
	r.m[1] = s;
	r.m[5] = c;
	return r;
}

// Перспективная проекция
// Камера смотрит вдоль -Z, глубина в спектре от 0 до й (где 0 - ближе).
// - поскольку у вулкана как и у питончика y направлена вниз
inline Mat4 mat_perspective(float fov_y, float aspect, float z_near, float z_far) {
	const float f = 1.0f / std::tan(fov_y / 2.0f);
	Mat4 r = {};
	r.m[0] = f / aspect;
	r.m[5] = -f;
	r.m[10] = z_far / (z_near - z_far);
	r.m[11] = -1.0f; 
	r.m[14] = z_near * z_far / (z_near - z_far);
	return r;
}


inline Mat4 mat_ortho(float half_h, float aspect, float z_near, float z_far) {
	const float half_w = half_h * aspect;
	Mat4 r = {};
	r.m[0] = 1.0f / half_w;
	r.m[5] = -1.0f / half_h;
	r.m[10] = -1.0f / (z_far - z_near);
	r.m[14] = -z_near / (z_far - z_near);
	r.m[15] = 1.0f;
	return r;
}
