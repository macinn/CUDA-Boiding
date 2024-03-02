#pragma once

constexpr auto fragment_box = R"(
#version 330 core

out vec4 fs_color;
void main()
{
    fs_color = vec4(1.0, 1.0, 1.0, 1.0);
}
)";

constexpr auto vertex_box = R"(
#version 330 core
layout(location = 0) in vec3 pos;

uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

void main()
{
    gl_Position = ProjectionMatrix * ViewMatrix * vec4(pos, 1.0);
}
)";