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
	RayShadow = 1,
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
	float3 gridOrigin;
	int3 gridSize;
};

struct LightSource
{
	float3 position;
	float3 power;
	float3 direction;
};

struct CameraRayData
{
	float3 position;
	float3 direction;
	int primIdx;
};

struct Photon
{
	float3 position;
	float3 energy;
	float3 dir;
	int primIdx;
};

struct PhotonHash
{
	Photon* pointer;
	int hashValue;

	bool operator < (const PhotonHash& a)
	{
		return hashValue < a.hashValue;
	}
};

struct PhotonPrd
{
	float3 energy;
	int startIdx;
	int numDeposits;	// number of times being recorded
	int depth;			// number of reflection
};

struct DebugData
{
	float3 v;
};

// data passed to Rt_RayGen
struct Rt_RayGenData
{
	CameraRayData* cameraRayDataLeft;
	CameraRayData* cameraRayDataRight;
};

struct Rt_HitData
{
	float3* normals;
	float3* kds;
	LightSource* lightSource;
	Photon* photonMap;
	int* NOLT;	// neighbour offset lookup table
	int* photonMapStartIdxs;
	CameraRayData* cameraRayDatasLeft;
	CameraRayData* cameraRayDatasRight;
};

#define PT_PHOTON_CNT ( 1 << 16 )
#define PT_MAX_DEPTH 8
#define PT_MAX_DEPOSIT 8

struct Pt_RayGenData
{
	LightSource* lightSource;
	Photon* photons;
};

struct Pt_HitData
{
	float3* normals;
	float3* kds;
	Photon* photons;
};

#define COLLECT_RAIDUS 0.02f
#define HASH_GRID_SIDELENGTH COLLECT_RAIDUS

#define hash(position) ((int)floorf((position.z - paras.gridOrigin.z) / HASH_GRID_SIDELENGTH)) * paras.gridSize.x * paras.gridSize.y \
+ ((int)floorf((position.y - paras.gridOrigin.y) / HASH_GRID_SIDELENGTH)) * paras.gridSize.x \
+ ((int)floorf((position.x - paras.gridOrigin.x) / HASH_GRID_SIDELENGTH))

#define BLOCK_SIZE 8
#define BLOCK_SIZE2 64

#define CUDA_GATHER
//#define OPTIX_GATHER

#define USE_SHARED_MEMORY

#define LIGHTDECAY 4.f

//#define USE_CONNECTRAY