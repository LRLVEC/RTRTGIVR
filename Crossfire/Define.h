#pragma once
#ifdef DefineDevice
#include <OptiX/_Define_7_Device.h>
/*struct TransInfo
{
	float3 row0; float blank0;
	float3 row1; float blank1;
	float3 row2; float blank2;
	float3 r0;
	float z0;
};*/
struct TransInfo
{
	//left eye
	float3 row0; float offset0Left;
	float3 row1; float offset1Left;
	float3 row2; float blank0;
	float3 r0Left; float z0;
	//right eye
	float offset0Right;
	float offset1Right;
	float eye;//0: left, 1: right
	float3 r0Right;
	//right eye proj
	float3 proj0Right; float ahh0;
	float3 proj1Right; float ahh1;
};
#else
using TransInfo = OpenGL::VR::OptiXTransCrossfire::TransInfo;
#endif
enum RayType
{
	RayRadiance = 0,
	RayOcclusion = 1,
	RayCount
};
struct RayData
{
	float r, g, b;
};
struct RayTraceData
{
	float3 answer;
	unsigned int depth;
	float3 albedo;
	float3 normal;//eye space
	float3 firstHitPos;
	float firstHitAngle;//dot(n, dir)
	float3 firstHitNorm;//world space
	int firstHitIdx;//-1 if no hit
};
struct CloseHitData
{
	float3* normals;
};
struct Parameters
{
	float4* imageLeft;
	float4* imageRight;
	float4* albedoLeft;
	float4* albedoRight;
	float4* normalLeft;
	float4* normalRight;
	OptixTraversableHandle handle;
	TransInfo* trans;
	uint2 size;
	cudaTextureObject_t cubeTexture;
	curandState* randState;
	unsigned int depthMax;
};