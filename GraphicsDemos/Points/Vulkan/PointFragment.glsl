#version 410 core
layout (location = 0) out vec4 fragColour;
layout (location = 1) in vec3 vertColour;

void main()
{
  fragColour = vec4(vertColour, 1.0);
}
