#version 400 core
in vec3 vertColour;
out vec4 fragColour;

void main()
{
  fragColour.rgb=vertColour;
}
