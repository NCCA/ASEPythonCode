#version 400 core
layout (location = 0) out vec4 fragColour;

in vec3 fragPos;
in vec3 normal;
in vec2 uv; // Not used, but passed from vertex shader

// Material properties
uniform vec3 objectColor;
uniform float specularStrength;
uniform float shininess;

// Light properties
uniform vec3 lightPos;
uniform vec3 lightColor;

// View properties
uniform vec3 viewPos;

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular (Blinn-Phong)
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    // Final color
    vec3 result = (ambient + diffuse + specular) * objectColor;
    fragColour = vec4(result, 1.0);
}
