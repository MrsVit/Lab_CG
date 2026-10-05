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


// Вершина конуса: положение и цвет
struct Vertex {
	float pos[3];
	float color[3];
};

struct ObjectData {
	Mat4 mvp;
	float tint[4];
};


struct Object {
	float pos[3];
	float rot[3];   // градусы
	float scale[3];
	float color[3]; // множитель цвета вершины
	bool follow;    // поддержка траектории
	float phase;    // сдвиг по времени вдоль траектории
};
//т.е куда поставить - как повернуть - размер - какого цвета - двигаться? - в какой момент?

constexpr int cone_segments = 48; // сколько сегментов в окружности основания (ghbvthyj 50)
constexpr float cone_radius = 0.5f;
constexpr float cone_height = 1.0f;

constexpr int max_objects = 8; //подобрано имперически, чтоб не висло

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

// Объекты на сцене
Object objects[max_objects];
int object_count = 2; //d yfxfkt dsdjlbv ldf rjyecf (jlby cnjzxbq? jlby ktnf.obq)
int selected = 0;

// Камера и проекция
bool perspective = true;
float fov = 60.0f;         // градусы
float ortho_half_h = 2.5f; //камера примерно в серединке конуса
float cam_dist = 4.0f;

// Анимация база
bool playing = true;
float speed = 1.0f;
float anim_time = 0.0f;
double last_time = 0.0;
float traj_radius = 1.5f;
float traj_height = 0.8f;
int freq_x = 3;
int freq_y = 2;
int freq_z = 1;
float spin = 90.0f; // поворот градусов в секунуду

// Объекты Vulkan
VkDescriptorSetLayout set_layout;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

VkBuffer vertex_buffer;
VmaAllocation vertex_alloc;
VkBuffer index_buffer;
VmaAllocation index_alloc;
uint32_t index_count;

// У каждого объекта свой uniform-буфер и свой набор дескрипторов (VkDescriptorSet)
VkDescriptorPool descriptor_pool;
VkDescriptorSet descriptor_sets[max_objects];
VkBuffer object_buffers[max_objects];
VmaAllocation object_allocs[max_objects];


// Начальные значения объекта с номером i
void initObject(int i) {
	Object& o = objects[i];
	for (int k = 0; k < 3; ++k) {
		o.pos[k] = 0.0f;
		o.rot[k] = 0.0f;
		o.scale[k] = 1.0f;
		o.color[k] = palette[i][k];
	}
	if (i == 0) {
		// главный конус стоит в центре, чуть наклонён, чтобы было видно основание
		for (int k = 0; k < 3; ++k) {
			o.scale[k] = 1.5f;
		}
		o.rot[0] = -20.0f;
		o.follow = false;
		o.phase = 0.0f;
	} else {
		// остальные - маленькие конусы, летящие 
		for (int k = 0; k < 3; ++k) {
			o.scale[k] = 0.5f;
		}o.follow = true;
		o.phase = float(i) * 1.5f;
	}
}

// Матрица модели: сначала растяжение, потом поворот, потом перенос
Mat4 makeModel(const Object& o) {
	float x = o.pos[0];
	float y = o.pos[1];
	float z = o.pos[2];
	float rx = o.rot[0];
	float ry = o.rot[1];
	float rz = o.rot[2];
	if (o.follow) {
		// Траектория - кривая Лиссажу в пространстве. Увидела на хабрике, красиво, вдобавок конус кувыркается.
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

	return mat_mul(move, mat_mul(rot, size));
}


void vertexColor(const float pos[3], float color[3]) {
	color[0] = pos[0] / (2.0f * cone_radius) + 0.5f;
	color[1] = pos[1] / cone_height + 0.5f;
	color[2] = pos[2] / (2.0f * cone_radius) + 0.5f;
}

void addVertex(std::vector<Vertex>& vertices, float x, float y, float z) {
	Vertex v = {};
	v.pos[0] = x;
	v.pos[1] = y;
	v.pos[2] = z;
	vertexColor(v.pos, v.color);
	vertices.push_back(v);
}


void buildCone(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const float half_h = cone_height / 2.0f;
	addVertex(vertices, 0.0f, half_h, 0.0f);
	for (int i = 0; i < cone_segments; ++i) {
		const float a = 2.0f * pi * float(i) / float(cone_segments);
		addVertex(vertices, cone_radius * std::cos(a), -half_h, cone_radius * std::sin(a));
	}
	addVertex(vertices, 0.0f, -half_h, 0.0f);
	const int apex = 0;
	const int center = cone_segments + 1;
	for (int i = 0; i < cone_segments; ++i) {
		const int cur = 1 + i;
		const int next = 1 + (i + 1) % cone_segments;
		// боковая грань (обход против часовой стрелки, если смотреть снаружи)
		indices.push_back(uint16_t(apex));
		indices.push_back(uint16_t(next));
		indices.push_back(uint16_t(cur));
		// треугольник основания (смотрит вниз)
		indices.push_back(uint16_t(center));
		indices.push_back(uint16_t(cur));
		indices.push_back(uint16_t(next));
	}
}

// Буфер в памяти, доступной процессору (данных мало, поэтому не использую промежуточный буфер)
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

bool createMesh() {
	auto& context = graphics::internal::context;
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
	buildCone(vertices, indices);
	index_count = uint32_t(indices.size());
	const VkDeviceSize vertices_size = vertices.size() * sizeof(Vertex);
	const VkDeviceSize indices_size = indices.size() * sizeof(uint16_t);
	if (!createBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  vertex_buffer, vertex_alloc)) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}
	if (!createBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                  index_buffer, index_alloc)) {
		std::cerr << "Failed to create index buffer\n";
		return false;
	}
	if (vmaCopyMemoryToAllocation(context.allocator, vertices.data(), vertex_alloc,
	                              0, vertices_size) != VK_SUCCESS ||
	    vmaCopyMemoryToAllocation(context.allocator, indices.data(), index_alloc,
	                              0, indices_size) != VK_SUCCESS) {
		std::cerr << "Failed to upload cone mesh\n";
		return false;
	}
	return true;
}

// Шаблон набора дескрипторов 
bool createDescriptors() {
	auto& context = graphics::internal::context;
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}
	for (int i = 0; i < max_objects; ++i) {
		if (!createBuffer(sizeof(ObjectData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  object_buffers[i], object_allocs[i])) {
			std::cerr << "Failed to create uniform buffer #" << i << '\n';
			return false;
		}
	}
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = max_objects,
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}
	const std::vector<VkDescriptorSetLayout> layouts(max_objects, set_layout);
	const VkDescriptorSetAllocateInfo sets_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = layouts.data(),
	};
	if (vkAllocateDescriptorSets(context.device, &sets_info,
	                             descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}
	// привязываем к каждому набору его uniform-буфер
	for (int i = 0; i < max_objects; ++i) {
		const VkDescriptorBufferInfo buffer_info = {
			.buffer = object_buffers[i],
			.offset = 0,
			.range = sizeof(ObjectData),
		};
		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};
		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	return true;
}

// Читает скомпилированный шейдер
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

bool createPipeline() {
	auto& context = graphics::internal::context;
	VkShaderModule vert = VK_NULL_HANDLE;
	VkShaderModule frag = VK_NULL_HANDLE;
	if (!loadShader("shaders/cone.vert.spv", vert) ||
	    !loadShader("shaders/cone.frag.spv", frag)) {
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
			.offset = offsetof(Vertex, color),
		},
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	// размеры viewport и scissor задаём при записи команд, чтобы не пересоздавать при изменении размера окна
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Отсечение граней выключено: конус выпуклый, а видимость решает буфер глубины.
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
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &set_layout,
	};
	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		vkDestroyShaderModule(context.device, vert, nullptr);
		vkDestroyShaderModule(context.device, frag, nullptr);
		return false;
	}
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
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &pipeline);

	// после создания пайплайна модули шейдеров уже не нужны, дропаем
	vkDestroyShaderModule(context.device, vert, nullptr);
	vkDestroyShaderModule(context.device, frag, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}


void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360.0f, 640.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 1 - Cone");
	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::RadioButton("Perspective", perspective)) {
			perspective = true;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !perspective)) {
			perspective = false;
		}
		ImGui::SliderFloat("Camera distance", &cam_dist, 1.0f, 15.0f);
		if (perspective) {
			ImGui::SliderFloat("FOV (deg)", &fov, 20.0f, 120.0f);
		} else {
			ImGui::SliderFloat("Ortho half-height", &ortho_half_h, 0.5f, 10.0f);
		}
	}
	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
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
		ImGui::ColorEdit3("Color", o.color);
		ImGui::Checkbox("Follow trajectory", &o.follow);
		ImGui::SliderFloat("Phase (rad)", &o.phase, 0.0f, 2.0f * pi);
		if (ImGui::Button("Reset object")) {
			initObject(selected);
		}
	}
	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
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

} 
// Интерфейс модуля
bool initialize() {
	for (int i = 0; i < object_count; ++i) {
		initObject(i);
	}
	if (!createMesh() || !createDescriptors() || !createPipeline()) {
		shutdown();
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr); // вместе с наборами внутри
	vkDestroyDescriptorSetLayout(context.device, set_layout, nullptr);
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	set_layout = VK_NULL_HANDLE;
	for (int i = 0; i < max_objects; ++i) {
		vmaDestroyBuffer(context.allocator, object_buffers[i], object_allocs[i]);
		object_buffers[i] = VK_NULL_HANDLE;
		object_allocs[i] = nullptr;
	}
	vmaDestroyBuffer(context.allocator, index_buffer, index_alloc);
	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_alloc);
	index_buffer = VK_NULL_HANDLE;
	index_alloc = nullptr;
	vertex_buffer = VK_NULL_HANDLE;
	vertex_alloc = nullptr;
}

void update(double time) {
	// Время анимации идёт только когда она не на паузе. Шаг ограничен, чтобы не рвалась
	float dt = float(time - last_time);
	last_time = time;
	if (dt < 0.0f) {
		dt = 0.0f;
	}if (dt > 0.1f) {
		dt = 0.1f;
	}if (playing) {
		anim_time += dt * speed;
	}drawInterface();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	VkCommandBuffer cmd = fd.command_buffer;
	if (cmd == VK_NULL_HANDLE) {
		return;
	}const uint32_t width = context.swapchain_extent.width;
	const uint32_t height = context.swapchain_extent.height;
	const float aspect = (height > 0) ? float(width) / float(height) : 1.0f;
	const Mat4 proj = perspective ? mat_perspective(to_rad(fov), aspect, 0.1f, 100.0f)
	                              : mat_ortho(ortho_half_h, aspect, 0.1f, 100.0f);
	const Mat4 view = mat_translate(0.0f, 0.0f, -cam_dist);
	const Mat4 view_proj = mat_mul(proj, view);

	for (int i = 0; i < object_count; ++i) {
		ObjectData data = {};
		data.mvp = mat_mul(view_proj, makeModel(objects[i]));
		data.tint[0] = objects[i].color[0];
		data.tint[1] = objects[i].color[1];
		data.tint[2] = objects[i].color[2];
		data.tint[3] = 1.0f;
		vmaCopyMemoryToAllocation(context.allocator, &data, object_allocs[i],
		                          0, sizeof(ObjectData));
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
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
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
	// все объекты используют одну и ту же геометрию конуса
	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT16);
	// а набор дескрипторов у каждого свой
	for (int i = 0; i < object_count; ++i) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &descriptor_sets[i], 0, nullptr);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}
	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

}
