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

	if (!paras.trans->eye)
	{
		curandState* statePtr = paras.randState + paras.size.x * index.y + index.x;
		curandStateMini state;
		getCurandState(&state, statePtr);
		RayData* rtData = (RayData*)optixGetSbtDataPointer();
		RayTraceData answer{ {0.f, 0.f, 0.f}, 0, {0.f, 0.f, 0.f},
			{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0, {0.f, 0.f, 0.f}, -1 };
		unsigned int pd0, pd1;
		pP(&answer, pd0, pd1);

		//left eye
		float2 ahh = /*random(index, paras.size, 0) +*/
			make_float2(index) - make_float2(paras.size) / 2.0f +
			make_float2(paras.trans->offset0Left, paras.trans->offset1Left);
		float3 d = normalize(make_float3(ahh, paras.trans->z0));
		float3 rayDir = make_float3(
			dot(paras.trans->row0, d),
			dot(paras.trans->row1, d),
			dot(paras.trans->row2, d));

		optixTrace(paras.handle, paras.trans->r0Left, rayDir,
			0.0001f, 1e16f,
			0.0f, OptixVisibilityMask(1),
			//OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
			OPTIX_RAY_FLAG_NONE,
			RayRadiance,        // SBT offset
			RayCount,           // SBT stride
			RayRadiance,        // missSBTIndex
			pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
		paras.imageLeft[index.y * paras.size.x + index.x] = make_float4(answer.answer, 0.f);
		paras.albedoLeft[index.y * paras.size.x + index.x] = make_float4(answer.albedo, 0.f);
		paras.normalLeft[index.y * paras.size.x + index.x] = make_float4(answer.normal, 0.f);

		if (answer.firstHitIdx != -1)
		{
			float3 dirRight = answer.firstHitPos - paras.trans->r0Right;
			float rMax(sqrtf(dot(dirRight, dirRight)));
			float3 dirRightEyeSpace(transposeMult(paras.trans->row0, paras.trans->row1, paras.trans->row2, dirRight));
			float2 dirRightScreenSpace{ dot(paras.trans->proj0Right,dirRightEyeSpace) / dirRightEyeSpace.z,
				dot(paras.trans->proj1Right,dirRightEyeSpace) / dirRightEyeSpace.z };
			dirRightScreenSpace = (1 - dirRightScreenSpace) / 2;
			uint2 idxRight{ dirRightScreenSpace.x * paras.size.x, dirRightScreenSpace.y * paras.size.y };
			if (dirRightScreenSpace.x <= 1 && dirRightScreenSpace.x >= -1 && dirRightScreenSpace.y <= 1 && dirRightScreenSpace.y >= -1)
			{
				dirRight /= rMax;
				float cosRight(dot(dirRight, answer.firstHitNorm));
				if (cosRight * answer.firstHitAngle > 0)
				{
					if (!atomicCAS((unsigned int*)&paras.imageRight[idxRight.y * paras.size.x + idxRight.x].w, 0, 0))
					{
						pd0 = 0;
						optixTrace(paras.handle, paras.trans->r0Right, dirRight,
							0.0001f, rMax, 0.0f, OptixVisibilityMask(1),
							OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
							RayOcclusion, RayCount, RayOcclusion, pd0);
						if (!pd0)
						{
							if (!atomicCAS((unsigned int*)&paras.imageRight[idxRight.y * paras.size.x + idxRight.x].w, 0, 1065353216))
							{
								//to do:
								//float coeff(answer.firstHitAngle * cosRight);
								paras.imageRight[idxRight.y * paras.size.x + idxRight.x] = make_float4(answer.answer, 1.f);
								paras.albedoRight[idxRight.y * paras.size.x + idxRight.x] = make_float4(answer.albedo, 0.f);
								paras.normalRight[idxRight.y * paras.size.x + idxRight.x] = make_float4(answer.normal, 0.f);
							}
						}
					}
				}
			}
		}
		setCurandState(statePtr, &state);
	}
	else
	{
		if (paras.imageRight[index.y * paras.size.x + index.x].w == 1.f)
			paras.imageRight[index.y * paras.size.x + index.x].w = 0;
		else
		{
			curandState* statePtr = paras.randState + paras.size.x * index.y + index.x;
			curandStateMini state;
			getCurandState(&state, statePtr);
			RayData* rtData = (RayData*)optixGetSbtDataPointer();
			RayTraceData answer{ {0.f, 0.f, 0.f}, 0, {0.f, 0.f, 0.f},
				{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0, {0.f, 0.f, 0.f}, -1 };
			unsigned int pd0, pd1;
			pP(&answer, pd0, pd1);

			//left eye
			float2 ahh = /*random(index, paras.size, 0) +*/
				make_float2(index) - make_float2(paras.size) / 2.0f +
				make_float2(paras.trans->offset0Right, paras.trans->offset1Right);
			float3 d = normalize(make_float3(ahh, paras.trans->z0));
			float3 rayDir = make_float3(
				dot(paras.trans->row0, d),
				dot(paras.trans->row1, d),
				dot(paras.trans->row2, d));

			optixTrace(paras.handle, paras.trans->r0Right, rayDir,
				0.0001f, 1e16f, 0.0f, OptixVisibilityMask(1),
				OPTIX_RAY_FLAG_NONE, RayRadiance, RayCount, RayRadiance,
				pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
			paras.imageRight[index.y * paras.size.x + index.x] = make_float4(answer.answer, 0.f);
			paras.albedoRight[index.y * paras.size.x + index.x] = make_float4(answer.albedo, 0.f);
			paras.normalRight[index.y * paras.size.x + index.x] = make_float4(answer.normal, 0.f);
			setCurandState(statePtr, &state);
		}
	}
}
extern "C" __global__ void __closesthit__Radiance()
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

		float3 color{ 0.1f, 0.9f, 0.9f };

		float3 rayDir(optixGetWorldRayDirection());
		float3 hitPoint(optixGetWorldRayOrigin() + rayDir * optixGetRayTmax());
		float cosi1 = dot(rayDir, n);
		if (ray->depth == 0)
		{
			ray->albedo = color;
			ray->normal = transposeMult(paras.trans->row0, paras.trans->row1, paras.trans->row2, n);
			ray->firstHitPos = hitPoint;
			ray->firstHitAngle = cosi1;
			ray->firstHitNorm = n;
			ray->firstHitIdx = primIdx;
		}

		//if (rayData.depth > russian)
		//{
		//	if (random(seed) < 0.2f) { rayData.color = answer; return; }
		//	else k /= 0.8f;
		//}
		if (cosi1 > 0) n = -n;
		unsigned int numRays(1);
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
			answer += ray->answer * color;
		}
		ray->answer = answer / numRays;
		setCurandStateToPayload(&state);
	}
	ray->depth -= 1;
}
extern "C" __global__ void __closesthit__Occlusion()
{
	optixSetPayload_0(1);
}
extern "C" __global__ void __miss__Radiance()
{
	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	RayTraceData* ray((RayTraceData*)uP(pd0, pd1));
	float3 dir(optixGetWorldRayDirection());
	float3 r(make_float3(texCubemap<float4>(paras.cubeTexture, dir.x, dir.y, dir.z)));
	if (ray->depth == 0)
	{
		ray->albedo = r;
	}
	ray->answer = r;
	ray->depth -= 1;
}