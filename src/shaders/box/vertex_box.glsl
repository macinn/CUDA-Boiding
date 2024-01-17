#version 330 core
layout(location = 0) in vec3 pos;

uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

void main()
{
    gl_Position = ProjectionMatrix * ViewMatrix * vec4(pos, 1.0);
}