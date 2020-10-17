#pragma once
#ifdef DefineDevice
#include <OptiX/_Define_7_Device.h>
struct TransInfo
{
	float3 row0; float blank0;
	float3 row1; float blank1;
	float3 row2; float blank2;
	float3 r0;
	float z0;
};
#else
using TransInfo = OpenGL::VR::OptiXTrans::TransInfo;
#endif
enum RayType
{
	RayRadiance = 0,
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
	float3 normal;
};
struct CloseHitData
{
	float3* normals;
};
struct Parameters
{
	float4* image;
	float4* albedo;
	float4* normal;
	OptixTraversableHandle handle;
	TransInfo* trans;
	uint2 size;
	cudaTextureObject_t cubeTexture;
	curandState* randState;
	unsigned int depthMax;
};