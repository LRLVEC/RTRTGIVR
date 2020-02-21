#define DefineDevice
#include "Define.h"
#include <texture_fetch_functions.h>
//To do: Construct again...

extern "C"
{
	__constant__ Parameters paras;
}

extern "C" __global__ void __raygen__RayAllocator()
{
	uint2 index = make_uint2(optixGetLaunchIndex());
	curandState* statePtr = paras.randState + paras.size.x * index.y + index.x;
	curandStateMini state;
	getCurandState(&state, statePtr);

	RayData* rtData = (RayData*)optixGetSbtDataPointer();
	RayTraceData answer = { {0.f,0.f,0.f},1 };

	float2 ahh = /*random(index, paras.size, 0) +*/
		make_float2(index) - make_float2(paras.size) / 2.0f +
		make_float2(paras.trans->blank0, paras.trans->blank1);
	float3 d = normalize(make_float3(ahh, paras.trans->z0));
	float3 dd = make_float3(
		dot(paras.trans->row0, d),
		dot(paras.trans->row1, d),
		dot(paras.trans->row2, d));
	unsigned int pd0, pd1;
	pP(&answer, pd0, pd1);
	optixTrace(paras.handle, paras.trans->r0, dd,
		0.0001f, 1e16f,
		0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
	paras.image[index.y * paras.size.x + index.x] = make_float4(answer.answer, 1.f);
	setCurandState(statePtr, &state);
}
extern "C" __global__ void __closesthit__Ahh()
{
	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	RayTraceData* ray((RayTraceData*)uP(pd0, pd1));
	if (ray->depth < paras.depthMax)
	{
		curandStateMini state(getCurandStateFromPayload());
		CloseHitData* closeHitData = (CloseHitData*)optixGetSbtDataPointer();
		int primIdx = optixGetPrimitiveIndex();
		float3 n = closeHitData->normals[primIdx];
		float3 answer{ 0 };
		//if (rayData.depth > russian)
		//{
		//	if (random(seed) < 0.2f) { rayData.color = answer; return; }
		//	else k /= 0.8f;
		//}
		float3 rayDir(optixGetWorldRayDirection());
		float3 hitPoint(optixGetWorldRayOrigin() + rayDir * optixGetRayTmax());
		float cosi1 = dot(rayDir, n);
		if (cosi1 > 0) n = -n;
		unsigned int numRays(10);
		for (int c0(0); c0 < numRays; ++c0)
		{
			ray->depth += 1;
			optixTrace(paras.handle, hitPoint, randomDirectionCosN(n, 1.0f, &state),
				0.0001f, 1e16f,
				0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
				RayRadiance,        // SBT offset
				RayCount,           // SBT stride
				RayRadiance,        // missSBTIndex
				pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
			answer += ray->answer * 0.8f;
		}
		ray->answer = answer / numRays;
		setCurandStateToPayload(&state);
	}
	ray->depth -= 1;
}
extern "C" __global__ void __miss__Ahh()
{
	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	RayTraceData* ray((RayTraceData*)uP(pd0, pd1));
	float3 dir(optixGetWorldRayDirection());
	ray->answer = make_float3(texCubemap<float4>(paras.cubeTexture, dir.x, dir.y, dir.z));
	ray->depth -= 1;
}