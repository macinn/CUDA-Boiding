#version 440

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_normal;
layout(location = 2) in vec3 vertex_offset;

out vec3 vs_position;
out vec3 vs_color;
out vec3 vs_normal;

uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

void main()
{
	vs_position = vec4(ModelMatrix * vec4(vertex_position + vertex_offset, 1.f)).xyz;
	vs_color = vec3(0.f, 1.f, 0.f);
	vs_normal = mat3(ModelMatrix) * vertex_normal;

	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vs_position, 1.f);

	//gl_Position = vec4(vertex_position + vertex_offset, 1.0);
}