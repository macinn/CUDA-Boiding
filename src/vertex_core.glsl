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

mat4 translate(vec3 translation) {
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(translation, 1.0)
    );
}
mat4 rotate(float angle, vec3 axis) {
    float c = cos(angle);
    float s = sin(angle);
    float ic = 1.0 - c;

    return mat4(
        vec4(ic * axis.x * axis.x + c, ic * axis.x * axis.y - s * axis.z, ic * axis.x * axis.z + s * axis.y, 0.0),
        vec4(ic * axis.x * axis.y + s * axis.z, ic * axis.y * axis.y + c, ic * axis.y * axis.z - s * axis.x, 0.0),
        vec4(ic * axis.x * axis.z - s * axis.y, ic * axis.y * axis.z + s * axis.x, ic * axis.z * axis.z + c, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
}

void main()
{
	vs_position = vec4(ModelMatrix * vec4(vertex_position + vertex_offset, 1.f)).xyz;
	vs_color = vec3(0.f, 1.f, 0.f);
	vs_normal = mat3(ModelMatrix) * vertex_normal;

	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vs_position, 1.f);

	//gl_Position = vec4(vertex_position + vertex_offset, 1.0);
}