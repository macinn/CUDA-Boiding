#version 440
#define eps 1e-8

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_normal;


layout(location = 2) in vec3 vertex_offset;
layout(location = 3) in vec3 vertex_velocity;

out vec3 vs_position;
out vec3 vs_color;
out vec3 vs_normal;

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
mat4 rotation3d(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = -sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(
        oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0.0,
        oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 0.0,
        oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}
mat4 createRotationMatrix(vec3 targetVector) {
    if (length(targetVector) < eps) {
		return mat4(1.0);
	}
	vec3 normalizedTarget = normalize(targetVector);
	vec3 axis = cross(vec3(0.0, 1.0, 0.0), normalizedTarget);
	float dotProductValue = dot(vec3(0.0, 1.0, 0.0), normalizedTarget);
	float angle = acos(dotProductValue);
    return rotation3d(axis, angle);
}

void main()
{
    mat4 ModelMatrix = translate(vertex_offset) * createRotationMatrix(vertex_velocity);
    vs_position = vec4(ModelMatrix * vec4(vertex_position, 1.f)).xyz;
	vs_color = vec3(0.f, 1.f, 0.f);
	vs_normal = mat3(ModelMatrix) * vertex_normal;

	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vs_position, 1.f);
    //gl_Position = vec4(vs_position, 1.f);
	//gl_Position = vec4(vertex_position + vertex_offset, 1.0);
}