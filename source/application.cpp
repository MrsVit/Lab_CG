#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "matrix.hpp"

namespace application {

namespace {

struct Vertex {
	float pos[3];
	float normal[3];
	float color[3];
};

struct SceneData {
	Mat4 view_proj;
	float camera_pos[4];
	float sun_dir[4];
	float sun_color[4];
	float ambient[4];
	int point_count;
	int spot_count;
	int pad[2];
};

static_assert(sizeof(SceneData) == 144, "SceneData must match the uniform block in cone.vert");

struct PointLightData {
	float position[4];
	float color[4];
};

struct SpotLightData {
	float position[4];
	float direction[4];
	float color[4];
	float angles[4];
};

static_assert(sizeof(PointLightData) == 32, "PointLightData must match PointLight in cone.frag");
static_assert(sizeof(SpotLightData) == 64, "SpotLightData must match SpotLight in cone.frag");

struct ModelData {
	Mat4 model;
	Mat4 normal_matrix;
	float albedo[4];
	float specular[4];
};

static_assert(sizeof(ModelData) == 160, "ModelData must match the uniform block in cone.vert");

struct UnlitPush {
	Mat4 mvp;
	float color[4];
};

static_assert(sizeof(UnlitPush) == 80, "UnlitPush must match the push constants in unlit.vert");

struct Object {
	float pos[3];
	float rot[3];
	float scale[3];
	float albedo[3];
	float specular[3];
	float shininess;
	bool vertex_colors;
	bool follow;
	float phase;
};

struct PointLight {
	float pos[3];
	float color[3];
	float intensity;
};

struct SpotLight {
	float pos[3];
	float dir[3];
	float color[3];
	float intensity;
	float inner;
	float outer;
};

constexpr int cone_segments = 48;
constexpr float cone_radius = 0.5f;
constexpr float cone_height = 1.0f;

constexpr int max_objects = 8;
constexpr int floor_model = max_objects;
constexpr int max_models = max_objects + 1;

constexpr int max_point_lights = 16;
constexpr int max_spot_lights = 16;

constexpr float marker_size = 0.15f;
constexpr float spot_cone_length = 2.5f;
constexpr int spot_ring_points = 16;
constexpr int spot_line_vertices = (spot_ring_points * 2 + 1) * 2;

const float palette[max_objects][3] = {
	{ 1.0f, 1.0f, 1.0f },
	{ 1.0f, 0.7f, 0.3f },
	{ 0.4f, 0.8f, 1.0f },
	{ 1.0f, 0.5f, 0.7f },
	{ 0.6f, 1.0f, 0.5f },
	{ 1.0f, 1.0f, 0.5f },
	{ 0.7f, 0.6f, 1.0f },
	{ 0.5f, 1.0f, 0.9f },
};

Object objects[max_objects];
int object_count = 2;
int selected = 0;

Object floor_obj;
bool show_floor = true;

bool perspective = true;
float fov = 60.0f;
float ortho_half_h = 2.5f;
float cam_pos[3] = { 0.0f, 1.5f, 6.5f };
float cam_yaw = 0.0f;
float cam_pitch = -10.0f;
float cam_speed = 4.0f;

float ambient_color[3] = { 0.30f, 0.32f, 0.40f };
float ambient_intensity = 0.35f;
bool sun_on = true;
float sun_dir[3] = { -0.4f, -1.0f, -0.3f };
float sun_color[3] = { 1.0f, 1.0f, 0.95f };
float sun_intensity = 0.4f;

PointLight point_lights[max_point_lights];
int point_count = 1;
int selected_point = 0;

SpotLight spot_lights[max_spot_lights];
int spot_count = 1;
int selected_spot = 0;

bool show_lights = true;

bool playing = true;
float speed = 1.0f;
float anim_time = 0.0f;
double last_time = 0.0;
float traj_radius = 1.5f;
float traj_height = 0.8f;
int freq_x = 3;
int freq_y = 2;
int freq_z = 1;
float spin = 90.0f;

struct Mesh {
	VkBuffer vertex_buffer;
	VmaAllocation vertex_alloc;
	VkBuffer index_buffer;
	VmaAllocation index_alloc;
	uint32_t index_count;
};

Mesh cone_mesh;
Mesh floor_mesh;
Mesh cube_mesh;

VkBuffer line_buffer;
VmaAllocation line_alloc;
uint32_t line_vertex_count;

VkDescriptorSetLayout scene_layout;
VkDescriptorSetLayout model_layout;
VkPipelineLayout pipeline_layout;
VkPipelineLayout unlit_layout;
VkPipeline pipeline;
VkPipeline unlit_pipeline;
VkPipeline line_pipeline;

VkDescriptorPool descriptor_pool;
VkDescriptorSet scene_set;
VkBuffer scene_buffer;
VmaAllocation scene_alloc;
VkBuffer point_buffer;
VmaAllocation point_alloc;
VkBuffer spot_buffer;
VmaAllocation spot_alloc;

VkDescriptorSet model_sets[max_models];
VkBuffer model_buffers[max_models];
VmaAllocation model_allocs[max_models];

void normalize3(const float v[3], float out[3]) {
	const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	if (len < 0.0001f) {
		out[0] = 0.0f;
		out[1] = -1.0f;
		out[2] = 0.0f;
		return;
	}

	out[0] = v[0] / len;
	out[1] = v[1] / len;
	out[2] = v[2] / len;
}

void cross3(const float a[3], const float b[3], float out[3]) {
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

void initObject(int i) {
	Object& o = objects[i];

	for (int k = 0; k < 3; ++k) {
		o.pos[k] = 0.0f;
		o.rot[k] = 0.0f;
		o.scale[k] = 1.0f;
		o.albedo[k] = palette[i][k];
		o.specular[k] = 0.6f;
	}

	o.shininess = 48.0f;
	o.vertex_colors = false;

	if (i == 0) {
		for (int k = 0; k < 3; ++k) {
			o.scale[k] = 1.5f;
		}
		o.rot[0] = -20.0f;
		o.follow = false;
		o.phase = 0.0f;
	} else {
		for (int k = 0; k < 3; ++k) {
			o.scale[k] = 0.5f;
		}
		o.follow = true;
		o.phase = float(i) * 1.5f;
	}
}

void initFloor() {
	for (int k = 0; k < 3; ++k) {
		floor_obj.pos[k] = 0.0f;
		floor_obj.rot[k] = 0.0f;
		floor_obj.scale[k] = 1.0f;
		floor_obj.albedo[k] = 0.55f;
		floor_obj.specular[k] = 0.25f;
	}

	floor_obj.pos[1] = -1.0f;
	floor_obj.scale[0] = 30.0f;
	floor_obj.scale[2] = 30.0f;
	floor_obj.albedo[2] = 0.62f;
	floor_obj.shininess = 24.0f;
	floor_obj.vertex_colors = false;
	floor_obj.follow = false;
	floor_obj.phase = 0.0f;
}

void initPointLight(int i) {
	PointLight& p = point_lights[i];

	const float a = 0.8f + 2.1f * float(i);
	p.pos[0] = 2.5f * std::cos(a);
	p.pos[1] = 0.8f;
	p.pos[2] = 2.5f * std::sin(a);

	for (int k = 0; k < 3; ++k) {
		p.color[k] = palette[(i + 1) % max_objects][k];
	}

	p.intensity = 4.0f;
}

void initSpotLight(int i) {
	SpotLight& s = spot_lights[i];

	const float side = (i % 2 == 0) ? -1.0f : 1.0f;
	s.pos[0] = side * 1.5f;
	s.pos[1] = 3.0f;
	s.pos[2] = 1.0f - 1.5f * float(i / 2);

	const float to_center[3] = { -s.pos[0], -s.pos[1], -s.pos[2] };
	normalize3(to_center, s.dir);

	for (int k = 0; k < 3; ++k) {
		s.color[k] = palette[(i * 3) % max_objects][k];
	}

	s.intensity = 10.0f;
	s.inner = 15.0f;
	s.outer = 28.0f;
}

void makeMatrices(const Object& o, Mat4& model, Mat4& normal_matrix) {
	float x = o.pos[0];
	float y = o.pos[1];
	float z = o.pos[2];
	float rx = o.rot[0];
	float ry = o.rot[1];
	float rz = o.rot[2];

	if (o.follow) {
		const float t = anim_time + o.phase;
		x += traj_radius * std::sin(float(freq_x) * t);
		y += traj_height * std::sin(float(freq_y) * t);
		z += traj_radius * std::cos(float(freq_z) * t);
		ry += spin * t;
		rx += spin * 0.5f * t;
	}

	const Mat4 move = mat_translate(x, y, z);
	const Mat4 rot = mat_mul(mat_rotate_z(to_rad(rz)),
	                         mat_mul(mat_rotate_y(to_rad(ry)), mat_rotate_x(to_rad(rx))));
	const Mat4 size = mat_scale(o.scale[0], o.scale[1], o.scale[2]);

	model = mat_mul(move, mat_mul(rot, size));
	normal_matrix = mat_mul(rot, mat_scale(1.0f / o.scale[0], 1.0f / o.scale[1], 1.0f / o.scale[2]));
}

Mat4 makeView() {
	const Mat4 move = mat_translate(-cam_pos[0], -cam_pos[1], -cam_pos[2]);
	const Mat4 turn = mat_mul(mat_rotate_x(to_rad(-cam_pitch)), mat_rotate_y(to_rad(-cam_yaw)));
	return mat_mul(turn, move);
}

void moveCamera(float dt) {
	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		cam_yaw -= io.MouseDelta.x * 0.2f;
		cam_pitch -= io.MouseDelta.y * 0.2f;

		if (cam_pitch > 89.0f) {
			cam_pitch = 89.0f;
		}
		if (cam_pitch < -89.0f) {
			cam_pitch = -89.0f;
		}
		if (cam_yaw > 180.0f) {
			cam_yaw -= 360.0f;
		}
		if (cam_yaw < -180.0f) {
			cam_yaw += 360.0f;
		}
	}

	if (io.WantCaptureKeyboard) {
		return;
	}

	const float yaw = to_rad(cam_yaw);
	const float pitch = to_rad(cam_pitch);

	const float fx = -std::sin(yaw) * std::cos(pitch);
	const float fy = std::sin(pitch);
	const float fz = -std::cos(yaw) * std::cos(pitch);
	const float rx = std::cos(yaw);
	const float rz = -std::sin(yaw);

	const float step = cam_speed * dt;

	if (ImGui::IsKeyDown(ImGuiKey_W)) {
		cam_pos[0] += fx * step;
		cam_pos[1] += fy * step;
		cam_pos[2] += fz * step;
	}
	if (ImGui::IsKeyDown(ImGuiKey_S)) {
		cam_pos[0] -= fx * step;
		cam_pos[1] -= fy * step;
		cam_pos[2] -= fz * step;
	}
	if (ImGui::IsKeyDown(ImGuiKey_D)) {
		cam_pos[0] += rx * step;
		cam_pos[2] += rz * step;
	}
	if (ImGui::IsKeyDown(ImGuiKey_A)) {
		cam_pos[0] -= rx * step;
		cam_pos[2] -= rz * step;
	}
	if (ImGui::IsKeyDown(ImGuiKey_E)) {
		cam_pos[1] += step;
	}
	if (ImGui::IsKeyDown(ImGuiKey_Q)) {
		cam_pos[1] -= step;
	}
}

void vertexColor(const float pos[3], float color[3]) {
	color[0] = pos[0] / (2.0f * cone_radius) + 0.5f;
	color[1] = pos[1] / cone_height + 0.5f;
	color[2] = pos[2] / (2.0f * cone_radius) + 0.5f;
}

void addVertex(std::vector<Vertex>& vertices, float x, float y, float z,
               float nx, float ny, float nz) {
	Vertex v = {};
	v.pos[0] = x;
	v.pos[1] = y;
	v.pos[2] = z;
	v.normal[0] = nx;
	v.normal[1] = ny;
	v.normal[2] = nz;
	vertexColor(v.pos, v.color);
	vertices.push_back(v);
}

void addPlainVertex(std::vector<Vertex>& vertices, float x, float y, float z,
                    float nx, float ny, float nz) {
	Vertex v = {};
	v.pos[0] = x;
	v.pos[1] = y;
	v.pos[2] = z;
	v.normal[0] = nx;
	v.normal[1] = ny;
	v.normal[2] = nz;
	v.color[0] = 1.0f;
	v.color[1] = 1.0f;
	v.color[2] = 1.0f;
	vertices.push_back(v);
}

void buildCone(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const float half_h = cone_height / 2.0f;
	const int n = cone_segments;

	const float slant = std::sqrt(cone_height * cone_height + cone_radius * cone_radius);
	const float n_side = cone_height / slant;
	const float n_up = cone_radius / slant;

	for (int i = 0; i < n; ++i) {
		const float a = 2.0f * pi * (float(i) + 0.5f) / float(n);
		addVertex(vertices, 0.0f, half_h, 0.0f,
		          n_side * std::cos(a), n_up, n_side * std::sin(a));
	}

	for (int i = 0; i < n; ++i) {
		const float a = 2.0f * pi * float(i) / float(n);
		addVertex(vertices, cone_radius * std::cos(a), -half_h, cone_radius * std::sin(a),
		          n_side * std::cos(a), n_up, n_side * std::sin(a));
	}

	addVertex(vertices, 0.0f, -half_h, 0.0f, 0.0f, -1.0f, 0.0f);

	for (int i = 0; i < n; ++i) {
		const float a = 2.0f * pi * float(i) / float(n);
		addVertex(vertices, cone_radius * std::cos(a), -half_h, cone_radius * std::sin(a),
		          0.0f, -1.0f, 0.0f);
	}

	const int center = 2 * n;

	for (int i = 0; i < n; ++i) {
		const int apex = i;
		const int cur = n + i;
		const int next = n + (i + 1) % n;

		indices.push_back(uint16_t(apex));
		indices.push_back(uint16_t(next));
		indices.push_back(uint16_t(cur));

		indices.push_back(uint16_t(center));
		indices.push_back(uint16_t(center + 1 + i));
		indices.push_back(uint16_t(center + 1 + (i + 1) % n));
	}
}

void buildFloor(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	addPlainVertex(vertices, -0.5f, 0.0f, -0.5f, 0.0f, 1.0f, 0.0f);
	addPlainVertex(vertices, -0.5f, 0.0f, 0.5f, 0.0f, 1.0f, 0.0f);
	addPlainVertex(vertices, 0.5f, 0.0f, 0.5f, 0.0f, 1.0f, 0.0f);
	addPlainVertex(vertices, 0.5f, 0.0f, -0.5f, 0.0f, 1.0f, 0.0f);

	const uint16_t list[] = { 0, 1, 2, 0, 2, 3 };
	for (uint16_t index : list) {
		indices.push_back(index);
	}
}

void buildCube(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	for (int i = 0; i < 8; ++i) {
		const float x = (i & 1) ? 0.5f : -0.5f;
		const float y = (i & 2) ? 0.5f : -0.5f;
		const float z = (i & 4) ? 0.5f : -0.5f;
		addPlainVertex(vertices, x, y, z, 0.0f, 0.0f, 0.0f);
	}

	const uint16_t list[] = {
		0, 2, 3, 0, 3, 1,
		4, 5, 7, 4, 7, 6,
		0, 4, 6, 0, 6, 2,
		1, 3, 7, 1, 7, 5,
		0, 1, 5, 0, 5, 4,
		2, 6, 7, 2, 7, 3,
	};
	for (uint16_t index : list) {
		indices.push_back(index);
	}
}

void addLine(std::vector<Vertex>& vertices, const float a[3], const float b[3]) {
	Vertex v = {};

	for (int k = 0; k < 3; ++k) {
		v.pos[k] = a[k];
	}
	vertices.push_back(v);

	for (int k = 0; k < 3; ++k) {
		v.pos[k] = b[k];
	}
	vertices.push_back(v);
}

void addSpotCone(std::vector<Vertex>& vertices, const SpotLight& s) {
	float dir[3];
	normalize3(s.dir, dir);

	float ref[3] = { 0.0f, 1.0f, 0.0f };
	if (std::fabs(dir[1]) > 0.99f) {
		ref[0] = 1.0f;
		ref[1] = 0.0f;
	}

	float right[3];
	cross3(dir, ref, right);
	normalize3(right, right);

	float up[3];
	cross3(right, dir, up);

	const float radius = spot_cone_length * std::tan(to_rad(s.outer));

	float center[3];
	for (int k = 0; k < 3; ++k) {
		center[k] = s.pos[k] + dir[k] * spot_cone_length;
	}

	float ring[spot_ring_points][3];
	for (int i = 0; i < spot_ring_points; ++i) {
		const float a = 2.0f * pi * float(i) / float(spot_ring_points);
		for (int k = 0; k < 3; ++k) {
			ring[i][k] = center[k] + radius * (right[k] * std::cos(a) + up[k] * std::sin(a));
		}
	}

	addLine(vertices, s.pos, center);

	for (int i = 0; i < spot_ring_points; ++i) {
		addLine(vertices, s.pos, ring[i]);
		addLine(vertices, ring[i], ring[(i + 1) % spot_ring_points]);
	}
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& alloc) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	return vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info,
	                       &buffer, &alloc, nullptr) == VK_SUCCESS;
}

bool createMesh(Mesh& mesh, const std::vector<Vertex>& vertices,
                const std::vector<uint16_t>& indices) {
	auto& context = graphics::internal::context;

	mesh.index_count = uint32_t(indices.size());

	const VkDeviceSize vertices_size = vertices.size() * sizeof(Vertex);
	const VkDeviceSize indices_size = indices.size() * sizeof(uint16_t);

	if (!createBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  mesh.vertex_buffer, mesh.vertex_alloc)) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}

	if (!createBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                  mesh.index_buffer, mesh.index_alloc)) {
		std::cerr << "Failed to create index buffer\n";
		return false;
	}

	if (vmaCopyMemoryToAllocation(context.allocator, vertices.data(), mesh.vertex_alloc,
	                              0, vertices_size) != VK_SUCCESS ||
	    vmaCopyMemoryToAllocation(context.allocator, indices.data(), mesh.index_alloc,
	                              0, indices_size) != VK_SUCCESS) {
		std::cerr << "Failed to upload mesh\n";
		return false;
	}

	return true;
}

void destroyMesh(Mesh& mesh) {
	auto& context = graphics::internal::context;

	vmaDestroyBuffer(context.allocator, mesh.index_buffer, mesh.index_alloc);
	vmaDestroyBuffer(context.allocator, mesh.vertex_buffer, mesh.vertex_alloc);
	mesh = {};
}

bool createMeshes() {
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;

	buildCone(vertices, indices);
	if (!createMesh(cone_mesh, vertices, indices)) {
		return false;
	}

	vertices.clear();
	indices.clear();
	buildFloor(vertices, indices);
	if (!createMesh(floor_mesh, vertices, indices)) {
		return false;
	}

	vertices.clear();
	indices.clear();
	buildCube(vertices, indices);
	return createMesh(cube_mesh, vertices, indices);
}

bool createLineBuffer() {
	const VkDeviceSize size = VkDeviceSize(max_spot_lights) * spot_line_vertices * sizeof(Vertex);

	if (!createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, line_buffer, line_alloc)) {
		std::cerr << "Failed to create line buffer\n";
		return false;
	}

	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &scene_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &model_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	if (!createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  scene_buffer, scene_alloc) ||
	    !createBuffer(sizeof(PointLightData) * max_point_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  point_buffer, point_alloc) ||
	    !createBuffer(sizeof(SpotLightData) * max_spot_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  spot_buffer, spot_alloc)) {
		std::cerr << "Failed to create scene buffers\n";
		return false;
	}

	for (int i = 0; i < max_models; ++i) {
		if (!createBuffer(sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  model_buffers[i], model_allocs[i])) {
			std::cerr << "Failed to create uniform buffer #" << i << '\n';
			return false;
		}
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 + max_models,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + max_models,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_set_info, &scene_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	const std::vector<VkDescriptorSetLayout> layouts(max_models, model_layout);

	const VkDescriptorSetAllocateInfo model_sets_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = max_models,
		.pSetLayouts = layouts.data(),
	};

	if (vkAllocateDescriptorSets(context.device, &model_sets_info,
	                             model_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate model descriptor sets\n";
		return false;
	}

	const VkDescriptorBufferInfo scene_infos[] = {
		{ .buffer = scene_buffer, .offset = 0, .range = sizeof(SceneData) },
		{ .buffer = point_buffer, .offset = 0, .range = sizeof(PointLightData) * max_point_lights },
		{ .buffer = spot_buffer, .offset = 0, .range = sizeof(SpotLightData) * max_spot_lights },
	};

	const VkWriteDescriptorSet scene_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_infos[0],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &scene_infos[1],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &scene_infos[2],
		},
	};

	vkUpdateDescriptorSets(context.device, sizeof(scene_writes) / sizeof(scene_writes[0]),
	                       scene_writes, 0, nullptr);

	for (int i = 0; i < max_models; ++i) {
		const VkDescriptorBufferInfo buffer_info = {
			.buffer = model_buffers[i],
			.offset = 0,
			.range = sizeof(ModelData),
		};

		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = model_sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};

		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	return true;
}

bool loadShader(const char* path, VkShaderModule& module) {
	auto& context = graphics::internal::context;

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader " << path
		          << " (run the app from the project root directory)\n";
		return false;
	}

	const size_t size = size_t(file.tellg());
	if (size == 0 || size % 4 != 0) {
		std::cerr << "Shader " << path << " is not a valid SPIR-V file\n";
		return false;
	}

	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module for " << path << '\n';
		return false;
	}

	return true;
}

bool makePipeline(VkPipelineLayout layout, const char* vert_path, const char* frag_path,
                  VkPrimitiveTopology topology, uint32_t attribute_count,
                  VkPipeline& result_pipeline) {
	auto& context = graphics::internal::context;

	VkShaderModule vert = VK_NULL_HANDLE;
	VkShaderModule frag = VK_NULL_HANDLE;

	if (!loadShader(vert_path, vert) || !loadShader(frag_path, frag)) {
		vkDestroyShaderModule(context.device, vert, nullptr);
		vkDestroyShaderModule(context.device, frag, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, pos),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = attribute_count,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};

	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic,
		.layout = layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &result_pipeline);

	vkDestroyShaderModule(context.device, vert, nullptr);
	vkDestroyShaderModule(context.device, frag, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

bool createPipelines() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayout set_layouts[] = { scene_layout, model_layout };

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(UnlitPush),
	};

	const VkPipelineLayoutCreateInfo unlit_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(context.device, &unlit_info, nullptr,
	                           &unlit_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout for unlit shapes\n";
		return false;
	}

	return makePipeline(pipeline_layout, "shaders/cone.vert.spv", "shaders/cone.frag.spv",
	                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, pipeline) &&
	       makePipeline(unlit_layout, "shaders/unlit.vert.spv", "shaders/unlit.frag.spv",
	                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 1, unlit_pipeline) &&
	       makePipeline(unlit_layout, "shaders/unlit.vert.spv", "shaders/unlit.frag.spv",
	                    VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 1, line_pipeline);
}

void drawMaterial(Object& o) {
	ImGui::ColorEdit3("Albedo", o.albedo);
	ImGui::ColorEdit3("Specular color", o.specular);
	ImGui::SliderFloat("Shininess", &o.shininess, 1.0f, 256.0f, "%.0f",
	                   ImGuiSliderFlags_Logarithmic);
	ImGui::Checkbox("Multiply by vertex colors", &o.vertex_colors);
}

void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 720.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 2 - Camera and lighting");

	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("camera");

		if (ImGui::RadioButton("Perspective", perspective)) {
			perspective = true;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !perspective)) {
			perspective = false;
		}

		ImGui::SliderFloat3("Position", cam_pos, -15.0f, 15.0f);
		ImGui::SliderFloat("Yaw (deg)", &cam_yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch (deg)", &cam_pitch, -89.0f, 89.0f);

		if (perspective) {
			ImGui::SliderFloat("FOV (deg)", &fov, 20.0f, 120.0f);
		} else {
			ImGui::SliderFloat("Ortho half-height", &ortho_half_h, 0.5f, 10.0f);
		}

		ImGui::SliderFloat("Move speed", &cam_speed, 0.5f, 15.0f);
		ImGui::TextWrapped("W/S/A/D - move, E/Q - up/down, "
		                   "hold right mouse button over the scene - look around");

		if (ImGui::Button("Reset camera")) {
			cam_pos[0] = 0.0f;
			cam_pos[1] = 1.5f;
			cam_pos[2] = 6.5f;
			cam_yaw = 0.0f;
			cam_pitch = -10.0f;
			fov = 60.0f;
		}

		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("lighting");

		ImGui::SeparatorText("Ambient light");
		ImGui::ColorEdit3("Ambient color", ambient_color);
		ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 1.0f);

		ImGui::SeparatorText("Directional light");
		ImGui::Checkbox("Enabled", &sun_on);
		ImGui::SliderFloat3("Direction", sun_dir, -1.0f, 1.0f);
		ImGui::ColorEdit3("Color", sun_color);
		ImGui::SliderFloat("Intensity", &sun_intensity, 0.0f, 5.0f);

		ImGui::SeparatorText("Visualization");
		ImGui::Checkbox("Show light markers", &show_lights);

		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("point");

		ImGui::Text("Lights: %d / %d", point_count, max_point_lights);

		if (ImGui::Button("Add")) {
			if (point_count < max_point_lights) {
				initPointLight(point_count);
				selected_point = point_count;
				point_count++;
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove selected")) {
			if (point_count > 0) {
				for (int i = selected_point; i < point_count - 1; ++i) {
					point_lights[i] = point_lights[i + 1];
				}
				point_count--;
				if (selected_point >= point_count) {
					selected_point = point_count - 1;
				}
				if (selected_point < 0) {
					selected_point = 0;
				}
			}
		}

		if (point_count > 1) {
			ImGui::SliderInt("Edit light", &selected_point, 0, point_count - 1);
		}

		if (point_count > 0) {
			PointLight& p = point_lights[selected_point];

			ImGui::SliderFloat3("Position", p.pos, -10.0f, 10.0f);
			ImGui::ColorEdit3("Color", p.color);
			ImGui::SliderFloat("Intensity", &p.intensity, 0.0f, 100.0f);
		}

		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("spot");

		ImGui::Text("Lights: %d / %d", spot_count, max_spot_lights);

		if (ImGui::Button("Add")) {
			if (spot_count < max_spot_lights) {
				initSpotLight(spot_count);
				selected_spot = spot_count;
				spot_count++;
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove selected")) {
			if (spot_count > 0) {
				for (int i = selected_spot; i < spot_count - 1; ++i) {
					spot_lights[i] = spot_lights[i + 1];
				}
				spot_count--;
				if (selected_spot >= spot_count) {
					selected_spot = spot_count - 1;
				}
				if (selected_spot < 0) {
					selected_spot = 0;
				}
			}
		}

		if (spot_count > 1) {
			ImGui::SliderInt("Edit light", &selected_spot, 0, spot_count - 1);
		}

		if (spot_count > 0) {
			SpotLight& s = spot_lights[selected_spot];

			ImGui::SliderFloat3("Position", s.pos, -10.0f, 10.0f);
			ImGui::SliderFloat3("Direction", s.dir, -1.0f, 1.0f);
			ImGui::ColorEdit3("Color", s.color);
			ImGui::SliderFloat("Intensity", &s.intensity, 0.0f, 100.0f);
			ImGui::SliderFloat("Inner angle (deg)", &s.inner, 0.0f, 80.0f);
			ImGui::SliderFloat("Outer angle (deg)", &s.outer, 1.0f, 80.0f);

			if (s.inner > s.outer) {
				s.inner = s.outer;
			}
		}

		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Objects")) {
		ImGui::Text("Objects on scene: %d / %d", object_count, max_objects);

		if (ImGui::Button("Add object")) {
			if (object_count < max_objects) {
				initObject(object_count);
				selected = object_count;
				object_count++;
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove last")) {
			if (object_count > 1) {
				object_count--;
				if (selected >= object_count) {
					selected = object_count - 1;
				}
			}
		}

		if (object_count > 1) {
			ImGui::SliderInt("Edit object", &selected, 0, object_count - 1);
		}

		Object& o = objects[selected];

		ImGui::SliderFloat3("Position", o.pos, -5.0f, 5.0f);
		ImGui::SliderFloat3("Rotation (deg)", o.rot, -180.0f, 180.0f);
		ImGui::SliderFloat3("Scale", o.scale, 0.1f, 3.0f);
		drawMaterial(o);
		ImGui::Checkbox("Follow trajectory", &o.follow);
		ImGui::SliderFloat("Phase (rad)", &o.phase, 0.0f, 2.0f * pi);

		if (ImGui::Button("Reset object")) {
			initObject(selected);
		}
	}

	if (ImGui::CollapsingHeader("Floor")) {
		ImGui::PushID("floor");

		ImGui::Checkbox("Show floor", &show_floor);
		ImGui::SliderFloat("Height", &floor_obj.pos[1], -5.0f, 1.0f);
		drawMaterial(floor_obj);

		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Animation")) {
		if (ImGui::Button(playing ? "Pause" : "Play")) {
			playing = !playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Reset time")) {
			anim_time = 0.0f;
		}

		ImGui::SliderFloat("Speed", &speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Trajectory radius", &traj_radius, 0.0f, 4.0f);
		ImGui::SliderFloat("Trajectory height", &traj_height, 0.0f, 3.0f);
		ImGui::SliderInt("Frequency X", &freq_x, 1, 6);
		ImGui::SliderInt("Frequency Y", &freq_y, 1, 6);
		ImGui::SliderInt("Frequency Z", &freq_z, 1, 6);
		ImGui::SliderFloat("Spin (deg/s)", &spin, 0.0f, 360.0f);
	}

	ImGui::End();
}

void writeModel(int index, const Object& o) {
	auto& context = graphics::internal::context;

	ModelData data = {};
	makeMatrices(o, data.model, data.normal_matrix);

	for (int k = 0; k < 3; ++k) {
		data.albedo[k] = o.albedo[k];
		data.specular[k] = o.specular[k];
	}
	data.albedo[3] = o.vertex_colors ? 1.0f : 0.0f;
	data.specular[3] = o.shininess;

	vmaCopyMemoryToAllocation(context.allocator, &data, model_allocs[index],
	                          0, sizeof(ModelData));
}

void writeScene(const Mat4& view_proj) {
	auto& context = graphics::internal::context;

	SceneData scene = {};
	scene.view_proj = view_proj;

	for (int k = 0; k < 3; ++k) {
		scene.camera_pos[k] = cam_pos[k];
		scene.sun_color[k] = sun_color[k];
		scene.ambient[k] = ambient_color[k];
	}
	normalize3(sun_dir, scene.sun_dir);
	scene.sun_color[3] = sun_on ? sun_intensity : 0.0f;
	scene.ambient[3] = ambient_intensity;
	scene.point_count = point_count;
	scene.spot_count = spot_count;

	vmaCopyMemoryToAllocation(context.allocator, &scene, scene_alloc, 0, sizeof(SceneData));

	PointLightData points[max_point_lights] = {};
	for (int i = 0; i < point_count; ++i) {
		for (int k = 0; k < 3; ++k) {
			points[i].position[k] = point_lights[i].pos[k];
			points[i].color[k] = point_lights[i].color[k];
		}
		points[i].color[3] = point_lights[i].intensity;
	}

	if (point_count > 0) {
		vmaCopyMemoryToAllocation(context.allocator, points, point_alloc,
		                          0, sizeof(PointLightData) * point_count);
	}

	SpotLightData spots[max_spot_lights] = {};
	for (int i = 0; i < spot_count; ++i) {
		normalize3(spot_lights[i].dir, spots[i].direction);
		for (int k = 0; k < 3; ++k) {
			spots[i].position[k] = spot_lights[i].pos[k];
			spots[i].color[k] = spot_lights[i].color[k];
		}
		spots[i].color[3] = spot_lights[i].intensity;
		spots[i].angles[0] = std::cos(to_rad(spot_lights[i].inner));
		spots[i].angles[1] = std::cos(to_rad(spot_lights[i].outer));
	}

	if (spot_count > 0) {
		vmaCopyMemoryToAllocation(context.allocator, spots, spot_alloc,
		                          0, sizeof(SpotLightData) * spot_count);
	}
}

void drawMarker(VkCommandBuffer cmd, const Mat4& view_proj, const float pos[3], const float color[3]) {
	UnlitPush push = {};
	push.mvp = mat_mul(view_proj, mat_mul(mat_translate(pos[0], pos[1], pos[2]),
	                                      mat_scale(marker_size, marker_size, marker_size)));
	push.color[0] = color[0];
	push.color[1] = color[1];
	push.color[2] = color[2];
	push.color[3] = 1.0f;

	vkCmdPushConstants(cmd, unlit_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	                   0, sizeof(UnlitPush), &push);
	vkCmdDrawIndexed(cmd, cube_mesh.index_count, 1, 0, 0, 0);
}

}

bool initialize() {
	for (int i = 0; i < object_count; ++i) {
		initObject(i);
	}
	initFloor();

	for (int i = 0; i < point_count; ++i) {
		initPointLight(i);
	}
	for (int i = 0; i < spot_count; ++i) {
		initSpotLight(i);
	}

	if (!createMeshes() || !createLineBuffer() || !createDescriptors() || !createPipelines()) {
		shutdown();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, line_pipeline, nullptr);
	vkDestroyPipeline(context.device, unlit_pipeline, nullptr);
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, unlit_layout, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr);
	line_pipeline = VK_NULL_HANDLE;
	unlit_pipeline = VK_NULL_HANDLE;
	pipeline = VK_NULL_HANDLE;
	unlit_layout = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	model_layout = VK_NULL_HANDLE;
	scene_layout = VK_NULL_HANDLE;

	for (int i = 0; i < max_models; ++i) {
		vmaDestroyBuffer(context.allocator, model_buffers[i], model_allocs[i]);
		model_buffers[i] = VK_NULL_HANDLE;
		model_allocs[i] = nullptr;
	}

	vmaDestroyBuffer(context.allocator, spot_buffer, spot_alloc);
	vmaDestroyBuffer(context.allocator, point_buffer, point_alloc);
	vmaDestroyBuffer(context.allocator, scene_buffer, scene_alloc);
	spot_buffer = VK_NULL_HANDLE;
	spot_alloc = nullptr;
	point_buffer = VK_NULL_HANDLE;
	point_alloc = nullptr;
	scene_buffer = VK_NULL_HANDLE;
	scene_alloc = nullptr;

	vmaDestroyBuffer(context.allocator, line_buffer, line_alloc);
	line_buffer = VK_NULL_HANDLE;
	line_alloc = nullptr;

	destroyMesh(cube_mesh);
	destroyMesh(floor_mesh);
	destroyMesh(cone_mesh);
}

void update(double time) {
	float dt = float(time - last_time);
	last_time = time;

	if (dt < 0.0f) {
		dt = 0.0f;
	}
	if (dt > 0.1f) {
		dt = 0.1f;
	}
	if (playing) {
		anim_time += dt * speed;
	}

	moveCamera(dt);
	drawInterface();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	VkCommandBuffer cmd = fd.command_buffer;
	if (cmd == VK_NULL_HANDLE) {
		return;
	}

	const uint32_t width = context.swapchain_extent.width;
	const uint32_t height = context.swapchain_extent.height;
	const float aspect = (height > 0) ? float(width) / float(height) : 1.0f;

	const Mat4 proj = perspective ? mat_perspective(to_rad(fov), aspect, 0.1f, 100.0f)
	                              : mat_ortho(ortho_half_h, aspect, 0.1f, 100.0f);
	const Mat4 view = makeView();
	const Mat4 view_proj = mat_mul(proj, view);

	writeScene(view_proj);

	for (int i = 0; i < object_count; ++i) {
		writeModel(i, objects[i]);
	}
	writeModel(floor_model, floor_obj);

	line_vertex_count = 0;
	if (show_lights && spot_count > 0) {
		std::vector<Vertex> lines;
		for (int i = 0; i < spot_count; ++i) {
			addSpotCone(lines, spot_lights[i]);
		}

		line_vertex_count = uint32_t(lines.size());
		vmaCopyMemoryToAllocation(context.allocator, lines.data(), line_alloc,
		                          0, lines.size() * sizeof(Vertex));
	}

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin);

	VkClearValue clear_values[2] = {};
	clear_values[0].color = {{ 0.08f, 0.09f, 0.12f, 1.0f }};
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(width),
		.height = float(height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = context.swapchain_extent,
	};
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_set, 0, nullptr);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &cone_mesh.vertex_buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, cone_mesh.index_buffer, 0, VK_INDEX_TYPE_UINT16);

	for (int i = 0; i < object_count; ++i) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &model_sets[i], 0, nullptr);
		vkCmdDrawIndexed(cmd, cone_mesh.index_count, 1, 0, 0, 0);
	}

	if (show_floor) {
		vkCmdBindVertexBuffers(cmd, 0, 1, &floor_mesh.vertex_buffer, &vertex_offset);
		vkCmdBindIndexBuffer(cmd, floor_mesh.index_buffer, 0, VK_INDEX_TYPE_UINT16);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &model_sets[floor_model], 0, nullptr);
		vkCmdDrawIndexed(cmd, floor_mesh.index_count, 1, 0, 0, 0);
	}

	if (show_lights) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, unlit_pipeline);
		vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh.vertex_buffer, &vertex_offset);
		vkCmdBindIndexBuffer(cmd, cube_mesh.index_buffer, 0, VK_INDEX_TYPE_UINT16);

		for (int i = 0; i < point_count; ++i) {
			drawMarker(cmd, view_proj, point_lights[i].pos, point_lights[i].color);
		}
		for (int i = 0; i < spot_count; ++i) {
			drawMarker(cmd, view_proj, spot_lights[i].pos, spot_lights[i].color);
		}

		if (line_vertex_count > 0) {
			UnlitPush push = {};
			push.mvp = view_proj;
			push.color[0] = 1.0f;
			push.color[1] = 1.0f;
			push.color[2] = 1.0f;
			push.color[3] = 1.0f;

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
			vkCmdBindVertexBuffers(cmd, 0, 1, &line_buffer, &vertex_offset);
			vkCmdPushConstants(cmd, unlit_layout,
			                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			                   0, sizeof(UnlitPush), &push);
			vkCmdDraw(cmd, line_vertex_count, 1, 0, 0);
		}
	}

	vkCmdEndRenderPass(cmd);

	vkEndCommandBuffer(cmd);
}

}