#version 450 core
layout(binding = 0)uniform sampler2D texSmp;
in vec2 texCood;
out vec4 o_color;
void main()
{
	vec4 ahh = texture(texSmp, texCood);
	o_color = /*vec4(0,1,0,0);*/ ahh / ahh.w;
}