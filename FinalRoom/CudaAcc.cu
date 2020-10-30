#define DefineDevice
#include <OptiX/_Define_7_Device.h>
#include "Define.h"

__global__ void initRandom(curandState* state, unsigned int seed, unsigned int MaxNum)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < MaxNum)
		curand_init(seed, id, 0, state + id);
}
void initRandom(curandState* state, int seed, unsigned int block, unsigned int grid, unsigned int MaxNum)
{
	initRandom << <grid, block >> > (state, seed, MaxNum);
}

__constant__ int NOLT[9];

__global__ void GatherKernelLeft(CameraRayData* cameraRayData, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, Parameters& paras)
{
	unsigned int idxLeft(blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x);
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	float3 hitPointPosition = cameraRayData[idxLeft].position;
	float3 hitPointDirection = cameraRayData[idxLeft].direction;
	int primIdx = cameraRayData[idxLeft].primIdx;
	float3 normal = normals[primIdx];
	float3 kd = kds[primIdx];

	if (primIdx == -1)return;
	// indirect flux
	int hitPointHashValue = hash(hitPointPosition);

	float3 indirectFlux = make_float3(0.0f, 0.0f, 0.0f);

	for (int c0(0); c0 < 9; c0++)
	{
		int gridNumber = hitPointHashValue + NOLT[c0];
		int startIdx = photonMapStartIdxs[gridNumber];
		int endIdx = photonMapStartIdxs[gridNumber + 3];
		for (int c1(startIdx); c1 < endIdx; c1++)
		{
			const Photon& photon = photonMap[c1];
			float3 diff = hitPointPosition - photon.position;
			float distance2 = dot(diff, diff);
			float photonDirDotNormal(dot(photon.dir, normal));

			if (distance2 <= COLLECT_RAIDUS * COLLECT_RAIDUS &&
				photonDirDotNormal * dot(hitPointDirection, normal) > 0)// && fabsf(dot(diff, normal)) < 0.0001f)
			{
				float Wpc = 1.0f - sqrtf(distance2) / COLLECT_RAIDUS;
				indirectFlux += photon.energy * kd * Wpc * fabsf(photonDirDotNormal);
			}
		}
	}
	indirectFlux /= M_PIf * COLLECT_RAIDUS * COLLECT_RAIDUS * (1.0f - 0.6667f / 1.0f) * PT_PHOTON_CNT;

	*(float3*)&paras.imageLeft[idxLeft] += indirectFlux;
#ifdef USE_Correlation
	int idxRight(*(int*)&paras.imageLeft[idxLeft].w);
	if (idxRight != -1)
	{
		*(float3*)&paras.imageRight[idxRight] += indirectFlux;
		//*(int*)&paras.imageRight[idxRight].w = -1;
	}
#endif
}

__global__ void GatherKernelRight(CameraRayData* cameraRayData, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, Parameters& paras)
{
	unsigned int idxRight(blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x);
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

#ifdef USE_Correlation
	int leftIndex(*(int*)&paras.imageRight[idxRight].w);
	if (leftIndex != -1)
	{
		*(int*)&paras.imageRight[idxRight].w = -1;
		return;
	}
#endif
	float3 hitPointPosition = cameraRayData[idxRight].position;
	float3 hitPointDirection = cameraRayData[idxRight].direction;
	int primIdx = cameraRayData[idxRight].primIdx;
	float3 normal = normals[primIdx];
	float3 kd = kds[primIdx];

	if (primIdx == -1)return;
	// indirect flux
	int hitPointHashValue = hash(hitPointPosition);

	float3 indirectFlux = make_float3(0.0f, 0.0f, 0.0f);

	for (int c0(0); c0 < 9; c0++)
	{
		int gridNumber = hitPointHashValue + NOLT[c0];
		int startIdx = photonMapStartIdxs[gridNumber];
		int endIdx = photonMapStartIdxs[gridNumber + 3];
		for (int c1(startIdx); c1 < endIdx; c1++)
		{
			const Photon& photon = photonMap[c1];
			float3 diff = hitPointPosition - photon.position;
			float distance2 = dot(diff, diff);
			float photonDirDotNormal(dot(photon.dir, normal));

			if (distance2 <= COLLECT_RAIDUS * COLLECT_RAIDUS &&
				photonDirDotNormal * dot(hitPointDirection, normal) > 0)// && fabsf(dot(diff, normal)) < 0.0001f)
			{
				float Wpc = 1.0f - sqrtf(distance2) / COLLECT_RAIDUS;
				indirectFlux += photon.energy * kd * Wpc * fabsf(photonDirDotNormal);
			}
		}
	}
	indirectFlux /= M_PIf * COLLECT_RAIDUS * COLLECT_RAIDUS * (1.0f - 0.6667f / 1.0f) * PT_PHOTON_CNT;
	*(float3*)&paras.imageRight[idxRight] += indirectFlux;
}


void initNOLT(int* NOLT_host)
{
	cudaMemcpyToSymbol(NOLT, NOLT_host, sizeof(NOLT));
}

void GatherLeft(CameraRayData* cameraRayDatas, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, uint2 size, Parameters& paras)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(size.x / dimBlock.x, size.y / dimBlock.y);
	GatherKernelLeft << <dimGrid, dimBlock >> > (cameraRayDatas, photonMap,
		normals, kds, photonMapStartIdxs, paras);
}

void GatherRight(CameraRayData* cameraRayDatas, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, uint2 size, Parameters& paras)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(size.x / dimBlock.x, size.y / dimBlock.y);
	GatherKernelRight << <dimGrid, dimBlock >> > (cameraRayDatas, photonMap,
		normals, kds, photonMapStartIdxs, paras);
}