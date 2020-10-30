#define DefineDevice
#include "Define.h"
#include <texture_fetch_functions.h>
//To do: Construct again...

extern "C"
{
	__constant__ Parameters paras;
}

extern "C" __global__ void __raygen__RayAllocatorLeft()
{
	uint2 indexLeft = make_uint2(optixGetLaunchIndex());
	unsigned int idxLeft(indexLeft.y * paras.size.x + indexLeft.x);
	Rt_RayGenData* raygenData = (Rt_RayGenData*)optixGetSbtDataPointer();
	CameraRayData& cameraRayDataLeft = raygenData->cameraRayDataLeft[idxLeft];
	cameraRayDataLeft.primIdx = -1;
	paras.imageLeft[idxLeft] = { 0 };

	float2 ahh = /*random(index, paras.size, 0) +*/
		make_float2(indexLeft) - make_float2(paras.size) / 2.0f +
		make_float2(paras.trans->offset0Left, paras.trans->offset1Left);
	float3 d = normalize(make_float3(ahh, paras.trans->z0));
	float3 rayDirLeft = make_float3(
		dot(paras.trans->row0, d),
		dot(paras.trans->row1, d),
		dot(paras.trans->row2, d));
	optixTrace(paras.handle, paras.trans->r0Left, rayDirLeft,
		0.0001f, 1e16f,
		0.0f, OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RayRadiance, RayCount, RayRadiance);
}
extern "C" __global__ void __raygen__RayAllocatorRight()
{
	uint2 indexRight = make_uint2(optixGetLaunchIndex());
	unsigned int idxRight(indexRight.y * paras.size.x + indexRight.x);
	//already written from left eye
	if (*(int*)&paras.imageRight[idxRight].w != -1)
		*(int*)&paras.imageRight[idxRight].w = -1;
	else
	{
		Rt_RayGenData* raygenData = (Rt_RayGenData*)optixGetSbtDataPointer();
		CameraRayData& cameraRayDataRight = raygenData->cameraRayDataRight[idxRight];
		cameraRayDataRight.primIdx = -1;
		*(float3*)&paras.imageRight[idxRight] = { 0 };
		float2 ahh = /*random(index, paras.size, 0) +*/
			make_float2(indexRight) - make_float2(paras.size) / 2.0f +
			make_float2(paras.trans->offset0Right, paras.trans->offset1Right);
		float3 d = normalize(make_float3(ahh, paras.trans->z0));
		float3 rayDirRight = make_float3(
			dot(paras.trans->row0, d),
			dot(paras.trans->row1, d),
			dot(paras.trans->row2, d));
		optixTrace(paras.handle, paras.trans->r0Right, rayDirRight,
			0.0001f, 1e16f,
			0.0f, OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_NONE,
			RayRadiance, RayCount, RayRadiance);
	}
}

extern "C" __global__ void __closesthit__RayRadianceLeft()
{
	int primIdx = optixGetPrimitiveIndex();
	uint2 indexLeft = make_uint2(optixGetLaunchIndex());
	unsigned int idxLeft(indexLeft.y * paras.size.x + indexLeft.x);

	Rt_HitData* hitData = (Rt_HitData*)optixGetSbtDataPointer();
	float3 rayDirLeft = optixGetWorldRayDirection();

	// hit point info
	float3 hitPointPosition = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirLeft;
	float3 hitPointKd = hitData->kds[primIdx];
	float3 hitPointNormal = hitData->normals[primIdx];

	float3 rayDirRight = hitPointPosition - paras.trans->r0Right;
	// record cameraRay info
	CameraRayData& cameraRayDataLeft = hitData->cameraRayDatasLeft[idxLeft];
	cameraRayDataLeft.position = hitPointPosition;
	cameraRayDataLeft.direction = rayDirLeft;
	cameraRayDataLeft.primIdx = primIdx;

	float rMax(sqrtf(dot(rayDirRight, rayDirRight)));
	float3 dirRightEyeSpace(transposeMult(paras.trans->row0, paras.trans->row1, paras.trans->row2, rayDirRight));
	float2 dirRightScreenSpace{ dot(paras.trans->proj0Right,dirRightEyeSpace) / dirRightEyeSpace.z,
		dot(paras.trans->proj1Right,dirRightEyeSpace) / dirRightEyeSpace.z };
	dirRightScreenSpace = (1 - dirRightScreenSpace) / 2;
	uint2 idxRight2{ dirRightScreenSpace.x * paras.size.x, dirRightScreenSpace.y * paras.size.y };
	unsigned int idxRight(idxRight2.y * paras.size.x + idxRight2.x);
	bool accessibleToRightEye(false);

	if (dirRightScreenSpace.x <= 1 && dirRightScreenSpace.x >= -1 && dirRightScreenSpace.y <= 1 && dirRightScreenSpace.y >= -1)
	{
		rayDirRight /= rMax;
		if (atomicCAS((int*)&paras.imageRight[idxRight].w, -1, -1) == -1)
		{
			unsigned int pd0(1);
			optixTrace(paras.handle, paras.trans->r0Right, rayDirRight,
				0.0001f, rMax, 0.0f, OptixVisibilityMask(1),
				OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				RayShadow, RayCount, RayShadow, pd0);
			if (pd0)
			{
				if (atomicCAS((int*)&paras.imageRight[idxRight].w, -1, idxLeft) == -1)
				{
					*(int*)&paras.imageLeft[idxLeft].w = idxRight;
					accessibleToRightEye = true;
					CameraRayData& cameraRayDataRight = hitData->cameraRayDatasRight[idxRight];
					cameraRayDataRight.position = hitPointPosition;
					cameraRayDataRight.direction = rayDirRight;
					cameraRayDataRight.primIdx = primIdx;
				}
			}
		}
	}

	// direct flux
	float3 lightSourcePosition;
	LightSource* lightSource = hitData->lightSource;
	lightSourcePosition = lightSource->position;
	float3 shadowRayDir = lightSourcePosition - hitPointPosition;
	float3 directFluxLeft = make_float3(0.0f, 0.0f, 0.0f);
	float3 directFluxRight = make_float3(0.0f, 0.0f, 0.0f);
	float rayDotNormalLeft(dot(rayDirLeft, hitPointNormal));
	float rayDotNormalRight(dot(rayDirRight, hitPointNormal));

	//different direct flux
	if (accessibleToRightEye && rayDotNormalLeft * rayDotNormalRight <= 0)
	{
		//for each light source
		float Tmax = sqrtf(dot(shadowRayDir, shadowRayDir));
		shadowRayDir = shadowRayDir / Tmax;
		float cosDN = dot(shadowRayDir, hitPointNormal);

		unsigned int pd0(1);
		optixTrace(paras.handle, hitPointPosition, shadowRayDir,
			0.0001f, Tmax,
			0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
			RayShadow, RayCount, RayShadow, pd0);
		if (pd0)
		{
			float3 dd(LIGHTDECAY * lightSource->power * hitPointKd * fabsf(cosDN));
			if (cosDN * rayDotNormalLeft < 0)directFluxLeft += dd;
			else directFluxRight += dd;
		}
	}
	//same direct flux
	else
	{
		//for each light source
		float cosDN = dot(shadowRayDir, hitPointNormal);
		if (cosDN * rayDotNormalLeft < 0)
		{
			float Tmax = sqrtf(dot(shadowRayDir, shadowRayDir));
			shadowRayDir = shadowRayDir / Tmax;
			float cosDN = dot(shadowRayDir, hitPointNormal);

			unsigned int pd0(1);
			optixTrace(paras.handle, hitPointPosition, shadowRayDir,
				0.0001f, Tmax,
				0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				RayShadow, RayCount, RayShadow, pd0);
			if (pd0)
			{
				float3 dd(LIGHTDECAY * lightSource->power * hitPointKd * fabsf(cosDN));
				directFluxLeft += dd;
			}
		}
		if (accessibleToRightEye)directFluxRight = directFluxLeft;
	}

	float3 screenSpaceNormal = transposeMult(paras.trans->row0, paras.trans->row1, paras.trans->row2, hitPointNormal);
	*(float3*)&paras.imageLeft[idxLeft] = directFluxLeft;
	paras.imageLeft[idxLeft] = make_float4(directFluxLeft, 0.f);
	paras.albedoLeft[idxLeft] = make_float4(hitPointKd, 0.f);
	paras.normalLeft[idxLeft] = make_float4(screenSpaceNormal, 0.f);
	if (accessibleToRightEye)
	{
		*(float3*)&paras.imageRight[idxRight] = directFluxRight;
		paras.albedoRight[idxRight] = make_float4(hitPointKd, 0.f);
		paras.normalRight[idxRight] = make_float4(screenSpaceNormal, 0.f);
	}
}
extern "C" __global__ void __closesthit__RayRadianceRight()
{
	int primIdx = optixGetPrimitiveIndex();
	uint2 index = make_uint2(optixGetLaunchIndex());
	unsigned int idxRight(index.y * paras.size.x + index.x);

	Rt_HitData* hitData = (Rt_HitData*)optixGetSbtDataPointer();
	float3 rayDirRight = optixGetWorldRayDirection();

	// hit point info
	float3 hitPointPosition = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirRight;
	float3 hitPointKd = hitData->kds[primIdx];
	float3 hitPointNormal = hitData->normals[primIdx];

	//record cameraRay info
	CameraRayData& cameraRayDataRight = hitData->cameraRayDatasRight[idxRight];
	cameraRayDataRight.position = hitPointPosition;
	cameraRayDataRight.direction = rayDirRight;
	cameraRayDataRight.primIdx = primIdx;

	// direct flux
	float3 lightSourcePosition;
	LightSource* lightSource = hitData->lightSource;
	lightSourcePosition = lightSource->position;
	float3 shadowRayDir = lightSourcePosition - hitPointPosition;
	float3 directFluxRight = make_float3(0.0f, 0.0f, 0.0f);
	float rayDotNormalRight(dot(rayDirRight, hitPointNormal));

	//for each light source
	float cosDN = dot(shadowRayDir, hitPointNormal);
	if (cosDN * rayDotNormalRight < 0)
	{
		float Tmax = sqrtf(dot(shadowRayDir, shadowRayDir));
		shadowRayDir = shadowRayDir / Tmax;
		float cosDN = dot(shadowRayDir, hitPointNormal);

		unsigned int pd0(1);
		optixTrace(paras.handle, hitPointPosition, shadowRayDir,
			0.0001f, Tmax,
			0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
			RayShadow, RayCount, RayShadow, pd0);
		if (pd0)
		{
			float3 dd(LIGHTDECAY * lightSource->power * hitPointKd * fabsf(cosDN));
			directFluxRight += dd;
		}
	}

	float3 screenSpaceNormal = transposeMult(paras.trans->row0, paras.trans->row1, paras.trans->row2, hitPointNormal);

	*(float3*)&paras.imageRight[idxRight] = directFluxRight;
	*(int*)&paras.imageRight[idxRight].w = -1;
	paras.albedoRight[idxRight] = make_float4(hitPointKd, 0.f);
	paras.normalRight[idxRight] = make_float4(screenSpaceNormal, 0.f);
}
extern "C" __global__ void __closesthit__Shadow()
{
	optixSetPayload_0(0);
}

// create a orthonormal basis from normalized vector n
static __device__ __inline__ void createOnb(const float3& n, float3& U, float3& V)
{
	U = cross(n, make_float3(0.0f, 1.0f, 0.0f));
	if (dot(U, U) < 1e-3)
		U = cross(n, make_float3(1.0f, 0.0f, 0.0f));
	U = normalize(U);
	V = cross(n, U);
}
extern "C" __global__ void __raygen__PhotonEmit()
{
	unsigned int index = optixGetLaunchIndex().x;
	int startIdx = index * PT_MAX_DEPOSIT;

	curandState* statePtr = paras.randState + index;
	curandStateMini state;
	getCurandState(&state, statePtr);

	Pt_RayGenData* raygenData = (Pt_RayGenData*)optixGetSbtDataPointer();
	LightSource* lightSource = raygenData->lightSource;

	// cos sampling(should be uniformly sampling a sphere)
	float3 dir = randomDirectionCosN(normalize(lightSource->direction), 1.0f, &state);
	float3 position = lightSource->position;

	/*float2 seed = curand_uniform2(&state);
	float z = 1 - 2 * seed.x;
	float r = sqrtf(fmax(0.0f, 1.0f - z * z));
	float phi = 2 * M_PIf * seed.y;
	float3 scale = make_float3(r * cosf(phi), r * sinf(phi), z);
	float3 lightDir = normalize(lightSource->direction);
	float3 U, V;
	createOnb(lightDir, U, V);

	position = lightSource->position;
	dir = normalize(lightDir * scale.z + U * scale.x + V * scale.y);*/

	// initialize photon records
	Photon* photons = raygenData->photons;
	for (int c0(0); c0 < PT_MAX_DEPOSIT; c0++)
		photons[startIdx + c0].energy = make_float3(0.0f);

	// set ray payload
	PhotonPrd prd;
	prd.energy = lightSource->power;
	prd.startIdx = startIdx;
	prd.numDeposits = 0;
	prd.depth = 0;
	unsigned int pd0, pd1;
	pP(&prd, pd0, pd1);

	// trace the photon
	optixTrace(paras.handle, position, dir,
		0.001f, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance, RayCount, RayRadiance,
		pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
	setCurandState(statePtr, &state);
}
extern "C" __global__ void __closesthit__PhotonHit()
{
	uint2 index = make_uint2(optixGetLaunchIndex());
	Pt_HitData* hitData = (Pt_HitData*)optixGetSbtDataPointer();
	int primIdx = optixGetPrimitiveIndex();

	curandStateMini state(getCurandStateFromPayload());

	// calculate the hit point
	float3 hitPointPosition = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
	float3 hitPointNormal = hitData->normals[primIdx];

	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	PhotonPrd prd = *(PhotonPrd*)uP(pd0, pd1);

	float3 newDir;
	float3 oldDir = optixGetWorldRayDirection();

	float3 kd = hitData->kds[primIdx];
	if (fmaxf(kd) > 0.0f)
	{
		// hit a diffuse surface; record hit if it has bounced at least once
		if (prd.depth > 0)
		{
			Photon& photon = hitData->photons[prd.startIdx + prd.numDeposits];
			photon.position = hitPointPosition;
			photon.energy = prd.energy;
			photon.dir = oldDir;
			photon.primIdx = primIdx;
			prd.numDeposits++;
		}

		// Russian roulette
		float Pd = fmaxf(kd);	// probability of being diffused
		if (curand_uniform(&state) > Pd)return;	// absorb
		prd.energy = kd * prd.energy / Pd;

		// cosine-weighted hemisphere sampling
		float3 W = { 0.f,0.f,0.f };
		if (dot(oldDir, hitPointNormal) > 0)W = -normalize(hitPointNormal);
		else W = normalize(hitPointNormal);
		newDir = randomDirectionCosN(W, 1.0f, &state);
	}

	prd.depth++;

	if (prd.numDeposits >= PT_MAX_DEPOSIT || prd.depth >= PT_MAX_DEPTH)
		return;

	pP(&prd, pd0, pd1);

	optixTrace(paras.handle, hitPointPosition, newDir,
		0.001f, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance, RayCount, RayRadiance,
		pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
}
