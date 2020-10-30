#include <cstdio>
#include <cstdlib>
#include <GL/_OpenGL.h>
#include <GL/_Window.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <OptiX/_OptiX_7.h>
#include <GL/_NBody.h>
#include "Define.h"
#include <_Time.h>
#include <_STL.h>
#include <_BMP.h>

void initRandom(curandState* state, int seed, unsigned int block, unsigned int grid, unsigned int MaxNum);
void initNOLT(int*);
void GatherLeft(CameraRayData* cameraRayDatas, Photon* photonMap, float3* normals, float3* kds, int* photonMapStartIdxs, uint2 size, Parameters& paras);
void GatherRight(CameraRayData* cameraRayDatas, Photon* photonMap, float3* normals, float3* kds, int* photonMapStartIdxs, uint2 size, Parameters& paras);


namespace CUDA
{
	namespace OptiX
	{
		struct PathTracing :OpenGL::VR::RayTracer
		{
			Context context;
			OptixModuleCompileOptions moduleCompileOptions;
			OptixPipelineCompileOptions pipelineCompileOptions;
			ModuleManager mm;
			OptixProgramGroupOptions programGroupOptions;
			Program rtRayAllocatorLeft;
			Program rtRayAllocatorRight;
			Program rtClosesthitRadianceLeft;
			Program rtClosesthitRadianceRight;
			Program rtClosesthitShadow;
			Program rtMissRadiance;
			Program rtMissShadow;
			Program ptRayAllocator;
			Program ptClosesthit;
			Program ptMiss;
			OptixPipelineLinkOptions rtPipelineLinkOptions;
			OptixPipelineLinkOptions ptPipelineLinkOptions;
			Pipeline rtPipLeft;
			Pipeline rtPipRight;
			Pipeline ptPip;
			SbtRecord<Rt_RayGenData> rt_raygenData[2];//2 pipelines
			SbtRecord<Rt_HitData> rtHitDatas[2][RayCount];//2 pipelines
			SbtRecord<Pt_RayGenData> pt_raygenData;
			SbtRecord<Pt_HitData> ptHitData;
			LightSource lightSource;
			Buffer lightSourceBuffer;
			Buffer cameraRayBufferLeft;
			Buffer cameraRayBufferRight;
			Buffer photonBuffer;
			Buffer photonMapBuffer;
			Buffer rtRaygenDataBufferLeft;
			Buffer rtRaygenDataBufferRight;
			Buffer rtHitDataBufferLeft;
			Buffer rtHitDataBufferRight;
			Buffer rtMissDataBuffer;
			Buffer ptRaygenDataBuffer;
			Buffer ptHitDataBuffer;
			Buffer ptMissDataBuffer;
			OptixShaderBindingTable rtSbtLeft;
			OptixShaderBindingTable rtSbtRight;
			OptixShaderBindingTable ptSbt;

			Buffer frameBufferLeft;
			Buffer frameBufferRight;
			Buffer frameAlbedoBufferLeft;
			Buffer frameAlbedoBufferRight;
			Buffer frameNormalBufferLeft;
			Buffer frameNormalBufferRight;
			Buffer finalFrameBufferLeft;
			Buffer finalFrameBufferRight;

			CUstream cuStream;
			Parameters paras;
			Buffer parasBuffer;
			STL box;
			Buffer vertices;
			Buffer normals;
			Buffer kds;
			Buffer NOLT;
			Buffer photonMapStartIdxs;
			OptixBuildInput triangleBuildInput;
			OptixAccelBuildOptions accelOptions;
			Buffer GASOutput;
			OptixTraversableHandle GASHandle;

			CubeMap sky;
			TextureCube texCube;
			//OptixDenoiserOptions denoiserOptions;
			//Denoiser denoiserLeft;
			//Denoiser denoiserRight;
			bool photonFlag;

			/*Buffer raygenDataBuffer;
			Buffer missDataBuffer;
			Buffer hitDataBuffer;
			OptixShaderBindingTable sbt;
			CUstream cuStream;
			Parameters paras;
			Buffer parasBuffer;
			STL box;
			Buffer vertices;
			Buffer normals;
			OptixBuildInput triangleBuildInput;
			OptixAccelBuildOptions accelOptions;
			Buffer GASOutput;
			OptixTraversableHandle GASHandle;
			CubeMap sky;
			TextureCube texCube;
			OptixDenoiserOptions denoiserOptions;
			Denoiser denoiserLeft;
			Denoiser denoiserRight;*/

			PathTracing(OpenGL::SourceManager* _sourceManager, OpenGL::FrameScale const& _size, void* transInfoDevice)
				:
				context(),
				moduleCompileOptions{ OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,OPTIX_COMPILE_OPTIMIZATION_DEFAULT,OPTIX_COMPILE_DEBUG_LEVEL_NONE },
				pipelineCompileOptions{ false,OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,8,2,OPTIX_EXCEPTION_FLAG_NONE,"paras",unsigned int(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE) },
				mm(&_sourceManager->folder, context, &moduleCompileOptions, &pipelineCompileOptions),
				programGroupOptions{},
				rtRayAllocatorLeft(Vector<String<char>>("__raygen__RayAllocatorLeft"), Program::RayGen, &programGroupOptions, context, &mm),
				rtRayAllocatorRight(Vector<String<char>>("__raygen__RayAllocatorRight"), Program::RayGen, &programGroupOptions, context, &mm),
				rtClosesthitRadianceLeft(Vector<String<char>>("__closesthit__RayRadianceLeft"), Program::HitGroup, &programGroupOptions, context, &mm),
				rtClosesthitRadianceRight(Vector<String<char>>("__closesthit__RayRadianceRight"), Program::HitGroup, &programGroupOptions, context, &mm),
				rtClosesthitShadow(Vector<String<char>>("__closesthit__Shadow"), Program::HitGroup, &programGroupOptions, context, &mm),
				rtMissRadiance(Vector<String<char>>(String<char>()), Program::Miss, &programGroupOptions, context, &mm),
				rtMissShadow(Vector<String<char>>(String<char>()), Program::Miss, &programGroupOptions, context, &mm),
				ptRayAllocator(Vector<String<char>>("__raygen__PhotonEmit"), Program::RayGen, &programGroupOptions, context, &mm),
				ptClosesthit(Vector<String<char>>("__closesthit__PhotonHit"), Program::HitGroup, &programGroupOptions, context, &mm),
				ptMiss(Vector<String<char>>(String<char>()), Program::Miss, &programGroupOptions, context, &mm),
				rtPipelineLinkOptions{ 1,OPTIX_COMPILE_DEBUG_LEVEL_NONE },
				ptPipelineLinkOptions{ 10,OPTIX_COMPILE_DEBUG_LEVEL_NONE },
				rtPipLeft(context, &pipelineCompileOptions, &rtPipelineLinkOptions, { rtRayAllocatorLeft, rtClosesthitRadianceLeft, rtClosesthitShadow, rtMissRadiance, rtMissShadow }),
				rtPipRight(context, &pipelineCompileOptions, &rtPipelineLinkOptions, { rtRayAllocatorRight, rtClosesthitRadianceRight, rtClosesthitShadow, rtMissRadiance, rtMissShadow }),
				ptPip(context, &pipelineCompileOptions, &ptPipelineLinkOptions, { ptRayAllocator, ptClosesthit, ptMiss }),

				lightSourceBuffer(lightSource, false),
				cameraRayBufferLeft(Buffer::Device),
				cameraRayBufferRight(Buffer::Device),
				photonBuffer(Buffer::Device),
				photonMapBuffer(Buffer::Device),
				rtRaygenDataBufferLeft(rt_raygenData[0], false),
				rtRaygenDataBufferRight(rt_raygenData[1], false),
				rtHitDataBufferLeft(Buffer::Device),
				rtHitDataBufferRight(Buffer::Device),
				rtMissDataBuffer(Buffer::Device),
				ptRaygenDataBuffer(pt_raygenData, false),
				ptHitDataBuffer(ptHitData, false),
				ptMissDataBuffer(Buffer::Device),
				rtSbtLeft({}),
				rtSbtRight({}),
				ptSbt({}),

				frameBufferLeft(CUDA::Buffer::Device),
				frameBufferRight(CUDA::Buffer::Device),
				frameAlbedoBufferLeft(CUDA::Buffer::Device),
				frameAlbedoBufferRight(CUDA::Buffer::Device),
				frameNormalBufferLeft(CUDA::Buffer::Device),
				frameNormalBufferRight(CUDA::Buffer::Device),
				finalFrameBufferLeft(CUDA::Buffer::GLinterop),
				finalFrameBufferRight(CUDA::Buffer::GLinterop),
				cuStream(0),
				parasBuffer(paras, false),
				box(_sourceManager->folder.find("resources/box.stl").readSTL()),
				vertices(Buffer::Device),
				normals(Buffer::Device),
				kds(Buffer::Device),
				NOLT(Buffer::Device),
				photonMapStartIdxs(Buffer::Device),

				triangleBuildInput({}),
				accelOptions({}),
				GASOutput(Buffer::Device),
				sky("resources/skybox/"),
				texCube({ 8, 8, 8, 8, cudaChannelFormatKindUnsigned }, cudaFilterModePoint, cudaReadModeNormalizedFloat, true, sky),
				//denoiserOptions{ OPTIX_DENOISER_INPUT_RGB },
				//denoiserLeft(context, { OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL }, OPTIX_DENOISER_MODEL_KIND_HDR, _size),
				//denoiserRight(context, { OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL }, OPTIX_DENOISER_MODEL_KIND_HDR, _size),
				photonFlag(true)
			{
				box.getVerticesRepeated();
				box.getNormals();
				box.printInfo(false);
				vertices.copy(box.verticesRepeated.data, sizeof(Math::vec3<float>)* box.verticesRepeated.length);
				normals.copy(box.normals.data, sizeof(Math::vec3<float>)* box.normals.length);

				paras.cubeTexture = texCube;

				float3* kdsTemp = new float3[box.normals.length];
				for (int c0(0); c0 < box.normals.length; c0++)
					kdsTemp[c0] = { 0.73f, 0.73f, 0.73f };
				kdsTemp[20] = make_float3(0.65f, 0.05f, 0.05f);
				kdsTemp[21] = make_float3(0.65f, 0.05f, 0.05f);
				kdsTemp[24] = make_float3(0.12f, 0.45f, 0.15f);
				kdsTemp[25] = make_float3(0.12f, 0.45f, 0.15f);
				//kdsTemp[box.normals.length - 6] = make_float3(0.65f, 0.05f, 0.05f);
				//kdsTemp[box.normals.length - 5] = make_float3(0.65f, 0.05f, 0.05f);
				//kdsTemp[box.normals.length - 10] = make_float3(0.12f, 0.45f, 0.15f);
				//kdsTemp[box.normals.length - 9] = make_float3(0.12f, 0.45f, 0.15f);
				kds.copy(kdsTemp, sizeof(float3) * box.normals.length);
				delete[] kdsTemp;

				lightSource.position = { 0.0f, 1.198f, -0.275f };
				float lightPower = 0.1f;
				lightSource.power = { lightPower, lightPower, lightPower };
				lightSource.direction = { 0.0f, -1.0f, 0.0f };
				lightSourceBuffer.copy(lightSource);

				// One per SBT record for this build input
				uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

				triangleBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				triangleBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				triangleBuildInput.triangleArray.vertexStrideInBytes = sizeof(Math::vec3<float>);
				triangleBuildInput.triangleArray.numVertices = box.verticesRepeated.length;
				triangleBuildInput.triangleArray.vertexBuffers = (CUdeviceptr*)&vertices.device;
				triangleBuildInput.triangleArray.flags = triangle_input_flags;
				triangleBuildInput.triangleArray.numSbtRecords = 1;
				triangleBuildInput.triangleArray.sbtIndexOffsetBuffer = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
				triangleBuildInput.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

				accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
				accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

				Buffer temp(Buffer::Device);
				Buffer compation(Buffer::Device);
				OptixAccelBufferSizes GASBufferSizes;
				optixAccelComputeMemoryUsage(context, &accelOptions, &triangleBuildInput, 1, &GASBufferSizes);
				temp.resize(GASBufferSizes.tempSizeInBytes);
				size_t compactedSizeOffset = ((GASBufferSizes.outputSizeInBytes + 7) / 8) * 8;
				compation.resize(compactedSizeOffset + 8);

				OptixAccelEmitDesc emitProperty = {};
				emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
				emitProperty.result = (CUdeviceptr)((char*)compation.device + compactedSizeOffset);

				optixAccelBuild(context, 0,
					&accelOptions, &triangleBuildInput, 1,// num build inputs, which is the num of vertexBuffers pointers
					temp, GASBufferSizes.tempSizeInBytes,
					compation, GASBufferSizes.outputSizeInBytes,
					&GASHandle, &emitProperty, 1);

				size_t compacted_gas_size;
				cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost);
				::printf("Compatcion: %u to %u\n", GASBufferSizes.outputSizeInBytes, compacted_gas_size);
				if (compacted_gas_size < GASBufferSizes.outputSizeInBytes)
				{
					GASOutput.resize(compacted_gas_size);
					// use handle as input and output-
					optixAccelCompact(context, 0, GASHandle, GASOutput, compacted_gas_size, &GASHandle);
				}
				else GASOutput.copy(compation);
				paras.handle = GASHandle;
				paras.trans = (TransInfo*)transInfoDevice;

				cameraRayBufferLeft.resize(sizeof(CameraRayData) * _size.w * _size.h);
				cameraRayBufferRight.resize(sizeof(CameraRayData) * _size.w * _size.h);

				// ray trace pass
				optixSbtRecordPackHeader(rtRayAllocatorLeft, &rt_raygenData[0]);
				optixSbtRecordPackHeader(rtRayAllocatorRight, &rt_raygenData[1]);
				rt_raygenData[0].data.cameraRayDataLeft = (CameraRayData*)cameraRayBufferLeft.device;
				rt_raygenData[0].data.cameraRayDataRight = (CameraRayData*)cameraRayBufferRight.device;
				rt_raygenData[1].data.cameraRayDataLeft = (CameraRayData*)cameraRayBufferLeft.device;
				rt_raygenData[1].data.cameraRayDataRight = (CameraRayData*)cameraRayBufferRight.device;
				rtRaygenDataBufferLeft.copy(rt_raygenData[0]);
				rtRaygenDataBufferRight.copy(rt_raygenData[1]);

				srand(time(nullptr));
				cudaMalloc(&paras.randState, _size.w* _size.h * sizeof(curandState));
				initRandom(paras.randState, rand(), 1024, (_size.w* _size.h + 1023) / 1024, _size.w* _size.h);

				/*OptixStackSizes stackSizes = { 0 };
				optixUtilAccumulateStackSizes(programGroups[0], &stackSizes);

				uint32_t max_trace_depth = 1;
				uint32_t max_cc_depth = 0;
				uint32_t max_dc_depth = 0;
				uint32_t direct_callable_stack_size_from_traversal;
				uint32_t direct_callable_stack_size_from_state;
				uint32_t continuation_stack_size;
				optixUtilComputeStackSizes(&stackSizes,
					max_trace_depth, max_cc_depth, max_dc_depth,
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state,
					&continuation_stack_size
				);
				optixPipelineSetStackSize(pipeline,
					direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state,
					continuation_stack_size, 3);*/

				optixSbtRecordPackHeader(rtClosesthitRadianceLeft, &rtHitDatas[0][RayRadiance]);
				optixSbtRecordPackHeader(rtClosesthitRadianceRight, &rtHitDatas[1][RayRadiance]);
				optixSbtRecordPackHeader(rtClosesthitShadow, &rtHitDatas[0][RayShadow]);
				optixSbtRecordPackHeader(rtClosesthitShadow, &rtHitDatas[1][RayShadow]);
				rtHitDatas[0][RayRadiance].data.normals = (float3*)normals.device;
				rtHitDatas[0][RayRadiance].data.kds = (float3*)kds.device;
				rtHitDatas[0][RayRadiance].data.lightSource = (LightSource*)lightSourceBuffer.device;
				rtHitDatas[0][RayRadiance].data.cameraRayDatasLeft = (CameraRayData*)cameraRayBufferLeft.device;
				rtHitDatas[0][RayRadiance].data.cameraRayDatasRight = (CameraRayData*)cameraRayBufferRight.device;
				rtHitDatas[0][RayShadow].data.normals = (float3*)normals.device;
				rtHitDatas[0][RayShadow].data.kds = (float3*)kds.device;
				rtHitDatas[0][RayShadow].data.lightSource = (LightSource*)lightSourceBuffer.device;
				rtHitDatas[0][RayShadow].data.cameraRayDatasLeft = (CameraRayData*)cameraRayBufferLeft.device;
				rtHitDatas[0][RayShadow].data.cameraRayDatasRight = (CameraRayData*)cameraRayBufferRight.device;
				//
				rtHitDatas[1][RayRadiance].data.normals = (float3*)normals.device;
				rtHitDatas[1][RayRadiance].data.kds = (float3*)kds.device;
				rtHitDatas[1][RayRadiance].data.lightSource = (LightSource*)lightSourceBuffer.device;
				rtHitDatas[1][RayRadiance].data.cameraRayDatasLeft = (CameraRayData*)cameraRayBufferLeft.device;
				rtHitDatas[1][RayRadiance].data.cameraRayDatasRight = (CameraRayData*)cameraRayBufferRight.device;
				rtHitDatas[1][RayShadow].data.normals = (float3*)normals.device;
				rtHitDatas[1][RayShadow].data.kds = (float3*)kds.device;
				rtHitDatas[1][RayShadow].data.lightSource = (LightSource*)lightSourceBuffer.device;
				rtHitDatas[1][RayShadow].data.cameraRayDatasLeft = (CameraRayData*)cameraRayBufferLeft.device;
				rtHitDatas[1][RayShadow].data.cameraRayDatasRight = (CameraRayData*)cameraRayBufferRight.device;

				rtHitDataBufferLeft.copy(rtHitDatas[0], sizeof(rtHitDatas[0]));
				rtHitDataBufferRight.copy(rtHitDatas[1], sizeof(rtHitDatas[1]));

				SbtRecord<int> rtMissDatas[2];
				optixSbtRecordPackHeader(rtMissRadiance, &rtMissDatas[0]);
				optixSbtRecordPackHeader(rtMissShadow, &rtMissDatas[1]);
				rtMissDataBuffer.copy(rtMissDatas, sizeof(rtMissDatas));

				rtSbtLeft.raygenRecord = rtRaygenDataBufferLeft;
				rtSbtLeft.hitgroupRecordBase = rtHitDataBufferLeft;
				rtSbtLeft.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Rt_HitData>);
				rtSbtLeft.hitgroupRecordCount = RayCount;
				rtSbtLeft.missRecordBase = rtMissDataBuffer;
				rtSbtLeft.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				rtSbtLeft.missRecordCount = RayCount;

				rtSbtRight.raygenRecord = rtRaygenDataBufferRight;
				rtSbtRight.hitgroupRecordBase = rtHitDataBufferRight;
				rtSbtRight.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Rt_HitData>);
				rtSbtRight.hitgroupRecordCount = RayCount;
				rtSbtRight.missRecordBase = rtMissDataBuffer;
				rtSbtRight.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				rtSbtRight.missRecordCount = RayCount;


				// photon trace pass
				photonBuffer.resize(sizeof(Photon) * PT_MAX_DEPOSIT * PT_PHOTON_CNT);

				srand(time(nullptr));
				cudaMalloc(&paras.randState, PT_PHOTON_CNT * sizeof(curandState));
				initRandom(paras.randState, rand(), 1024, (PT_PHOTON_CNT + 1023) / 1024, PT_PHOTON_CNT);

				optixSbtRecordPackHeader(ptRayAllocator, &pt_raygenData);
				pt_raygenData.data.lightSource = (LightSource*)lightSourceBuffer.device;
				pt_raygenData.data.photons = (Photon*)photonBuffer.device;
				ptRaygenDataBuffer.copy(pt_raygenData);

				optixSbtRecordPackHeader(ptClosesthit, &ptHitData);
				ptHitData.data.normals = (float3*)normals.device;
				ptHitData.data.kds = (float3*)kds.device;
				ptHitData.data.photons = (Photon*)photonBuffer.device;
				ptHitDataBuffer.copy(ptHitData);

				SbtRecord<int> pt_missData;
				optixSbtRecordPackHeader(ptMiss, &pt_missData);
				ptMissDataBuffer.copy(pt_missData);

				ptSbt.raygenRecord = ptRaygenDataBuffer;
				ptSbt.hitgroupRecordBase = ptHitDataBuffer;
				ptSbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Pt_HitData>);
				ptSbt.hitgroupRecordCount = 1;
				ptSbt.missRecordBase = ptMissDataBuffer;
				ptSbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				ptSbt.missRecordCount = 1;

				cudaStreamCreate(&cuStream);
				//denoiser.setup(cuStream);

			}
			virtual void run()
			{
				if (photonFlag)
				{
					optixLaunch(ptPip, cuStream, parasBuffer, sizeof(Parameters), &ptSbt, PT_PHOTON_CNT, 1, 1);
					createPhotonMap();
					photonFlag = false;
				}
				
				optixLaunch(rtPipLeft, cuStream, parasBuffer, sizeof(Parameters), &rtSbtLeft, paras.size.x, paras.size.y, 1);
				optixLaunch(rtPipRight, cuStream, parasBuffer, sizeof(Parameters), &rtSbtRight, paras.size.x, paras.size.y, 1);
				
				GatherLeft((CameraRayData*)cameraRayBufferLeft.device, (Photon*)photonMapBuffer.device,
					(float3*)normals.device, (float3*)kds.device, (int*)photonMapStartIdxs.device,
					paras.size, *(Parameters*)parasBuffer.device);
				GatherRight((CameraRayData*)cameraRayBufferRight.device, (Photon*)photonMapBuffer.device,
					(float3*)normals.device, (float3*)kds.device, (int*)photonMapStartIdxs.device,
					paras.size, *(Parameters*)parasBuffer.device);

				//finalFrameBufferLeft.map();
				//denoiserLeft.run(cuStream);
				//finalFrameBufferLeft.unmap();
				//finalFrameBufferRight.map();
				//denoiserRight.run(cuStream);
				//finalFrameBufferRight.unmap();
			}
			virtual void resize(OpenGL::FrameScale const& _size, GLuint _glLeft, GLuint _glRight)
			{
				size_t frameSize(sizeof(float4) * _size.w * _size.h);
				frameBufferLeft.resize(frameSize);
				frameBufferRight.resize(frameSize);
				frameAlbedoBufferLeft.resize(frameSize);
				frameAlbedoBufferRight.resize(frameSize);
				frameNormalBufferLeft.resize(frameSize);
				frameNormalBufferRight.resize(frameSize);
				finalFrameBufferLeft.resize(_glLeft);
				finalFrameBufferRight.resize(_glRight);
				finalFrameBufferLeft.map();
				finalFrameBufferRight.map();
				
				//cameraRayBufferLeft.resize(sizeof(CameraRayData) * _size.w * _size.h);
				//cameraRayBufferRight.resize(sizeof(CameraRayData) * _size.w * _size.h);
				//rt_raygenData.data.cameraRayDataLeft = (CameraRayData*)cameraRayBufferLeft.device;
				//rt_raygenData.data.cameraRayDataRight = (CameraRayData*)cameraRayBufferRight.device;
				//rtRaygenDataBuffer.copy(rt_raygenData);
				//rtHitDatas[0][RayRadiance].data.cameraRayDatasLeft = (CameraRayData*)cameraRayBufferLeft.device;
				//rtHitDatas[0][RayShadow].data.cameraRayDatasLeft = (CameraRayData*)cameraRayBufferLeft.device;
				//rtHitDatas[1][RayRadiance].data.cameraRayDatasRight = (CameraRayData*)cameraRayBufferRight.device;
				//rtHitDatas[1][RayShadow].data.cameraRayDatasRight = (CameraRayData*)cameraRayBufferRight.device;
				//rtHitDataBufferLeft.copy(rtHitDatas[0], sizeof(rtHitDatas[0]));
				//rtHitDataBufferRight.copy(rtHitDatas[1], sizeof(rtHitDatas[1]));

				//denoiserLeft.setup((float*)frameBufferLeft.device, (float*)frameAlbedoBufferLeft.device,
				//	(float*)frameNormalBufferLeft.device, (float*)finalFrameBufferLeft.device, cuStream);
				//denoiserRight.setup((float*)frameBufferRight.device, (float*)frameAlbedoBufferRight.device,
				//	(float*)frameNormalBufferRight.device, (float*)finalFrameBufferRight.device, cuStream);

				paras.imageLeft = (float4*)finalFrameBufferLeft.device;
				paras.imageRight = (float4*)finalFrameBufferRight.device;
				paras.albedoLeft = (float4*)frameAlbedoBufferLeft.device;
				paras.albedoRight = (float4*)frameAlbedoBufferRight.device;
				paras.normalLeft = (float4*)frameNormalBufferLeft.device;
				paras.normalRight = (float4*)frameNormalBufferRight.device;
				paras.size = make_uint2(_size.w, _size.h);
				parasBuffer.copy(paras);
				finalFrameBufferLeft.unmap();
				finalFrameBufferRight.unmap();
			}
			void createPhotonMap()
			{
				// copy photon data to host
				size_t photonBufferSize = photonBuffer.size;
				int photonBufferCnt = photonBufferSize / sizeof(Photon);
				Photon* photonData = new Photon[photonBufferCnt];
				cudaMemcpy(photonData, photonBuffer.device, photonBufferSize, cudaMemcpyDeviceToHost);

				// get valid data and compute bounding box
				float floatMax = std::numeric_limits<float>::max();
				float3 bbmin = make_float3(floatMax, floatMax, floatMax);
				float3 bbmax = make_float3(-floatMax, -floatMax, -floatMax);

				int validPhotonCnt = 0;
				PhotonHash* tempPhotons = new PhotonHash[photonBufferCnt];
				for (int c0(0); c0 < photonBufferCnt; c0++)
					if (photonData[c0].energy.x > 0.0f ||
						photonData[c0].energy.y > 0.0f ||
						photonData[c0].energy.z > 0.0f)
					{
						float3 position = photonData[c0].position;

						tempPhotons[validPhotonCnt++].pointer = &photonData[c0];

						bbmin.x = fminf(bbmin.x, position.x);
						bbmin.y = fminf(bbmin.y, position.y);
						bbmin.z = fminf(bbmin.z, position.z);
						bbmax.x = fmaxf(bbmax.x, position.x);
						bbmax.y = fmaxf(bbmax.y, position.y);
						bbmax.z = fmaxf(bbmax.z, position.z);
					}

				printf("photonBufferCnt: %d, valid cnt: %d\n", photonBufferCnt, validPhotonCnt);

				bbmin.x -= 0.001f;
				bbmin.y -= 0.001f;
				bbmin.z -= 0.001f;
				bbmax.x += 0.001f;
				bbmax.y += 0.001f;
				bbmax.z += 0.001f;

				/*printf("bbmin:(%f,%f,%f)\n", bbmin.x, bbmin.y, bbmin.z);
				printf("bbmax:(%f,%f,%f)\n", bbmax.x, bbmax.y, bbmax.z);*/

				// specify the grid size
				paras.gridSize.x = (int)ceilf((bbmax.x - bbmin.x) / HASH_GRID_SIDELENGTH) + 2;
				paras.gridSize.y = (int)ceilf((bbmax.y - bbmin.y) / HASH_GRID_SIDELENGTH) + 2;
				paras.gridSize.z = (int)ceilf((bbmax.z - bbmin.z) / HASH_GRID_SIDELENGTH) + 2;
				/*printf("gridSize:(%d,%d,%d)\n", paras.gridSize.x, paras.gridSize.y, paras.gridSize.z);*/

				// specify the world origin
				paras.gridOrigin.x = bbmin.x - HASH_GRID_SIDELENGTH;
				paras.gridOrigin.y = bbmin.y - HASH_GRID_SIDELENGTH;
				paras.gridOrigin.z = bbmin.z - HASH_GRID_SIDELENGTH;
				/*printf("gridOrigin:(%f,%f,%f)\n", paras.gridOrigin.x, paras.gridOrigin.y, paras.gridOrigin.z);
*/
				parasBuffer.copy(paras);

				// compute hash value
				for (int c0(0); c0 < validPhotonCnt; c0++)
					tempPhotons[c0].hashValue = hash(tempPhotons[c0].pointer->position);

				// sort according to hash value
				qsort(tempPhotons, 0, validPhotonCnt);

				// create neighbour offset lookup table
				int* NOLTDatas = new int[9];
				float3 offset[9] = { {-1,-1,-1},{-1,0,-1},{-1,1,-1},{-1,-1,0},{-1,0,0},{-1,1,0},{-1,-1,1},{-1,0,1},{-1,1,1} };
				for (int c0(0); c0 < 9; c0++)
					NOLTDatas[c0] = offset[c0].z * paras.gridSize.x * paras.gridSize.y + offset[c0].y * paras.gridSize.x + offset[c0].x;

				initNOLT(NOLTDatas);
				delete[] NOLTDatas;

				// reorder to build the photonMap
				Photon* photonMapData = new Photon[validPhotonCnt];
				for (int c0(0); c0 < validPhotonCnt; c0++)
					photonMapData[c0] = *(tempPhotons[c0].pointer);
				photonMapBuffer.copy(photonMapData, sizeof(Photon) * validPhotonCnt);

				// find the start index for each cell
				int gridCnt = paras.gridSize.x * paras.gridSize.y * paras.gridSize.z;
				int* startIdxs = new int[gridCnt + 1];
				int i = 0;	// for startIdxs
				int j = 0;	// for tempPhotons
				while (i <= gridCnt)
				{
					if (i < tempPhotons[j].hashValue)
					{
						startIdxs[i++] = j;
						continue;
					}
					if (i == tempPhotons[j].hashValue)
					{
						startIdxs[i] = j;
						while (i == tempPhotons[j].hashValue && j < validPhotonCnt)
							j++;
						i++;
					}
					if (j == validPhotonCnt)
					{
						while (i <= gridCnt)
							startIdxs[i++] = j;
					}
				}
				photonMapStartIdxs.copy(startIdxs, sizeof(int) * (gridCnt + 1));

				/*int emptyCnt = 0;
				for (int i = 0; i < gridCnt; i++)
					if (startIdxs[i] == startIdxs[i + 1])
						emptyCnt++;
				printf("empty %f\%\n", (float)emptyCnt / gridCnt);*/

				delete[] startIdxs;

				// update the sbt
				rtHitDatas[0][RayRadiance].data.photonMap = (Photon*)photonMapBuffer.device;
				rtHitDatas[0][RayRadiance].data.photonMapStartIdxs = (int*)photonMapStartIdxs.device;
				rtHitDatas[0][RayRadiance].data.NOLT = (int*)NOLT.device;
				rtHitDatas[0][RayShadow].data.photonMap = (Photon*)photonMapBuffer.device;
				rtHitDatas[0][RayShadow].data.photonMapStartIdxs = (int*)photonMapStartIdxs.device;
				rtHitDatas[0][RayShadow].data.NOLT = (int*)NOLT.device;

				rtHitDatas[1][RayRadiance].data.photonMap = (Photon*)photonMapBuffer.device;
				rtHitDatas[1][RayRadiance].data.photonMapStartIdxs = (int*)photonMapStartIdxs.device;
				rtHitDatas[1][RayRadiance].data.NOLT = (int*)NOLT.device;
				rtHitDatas[1][RayShadow].data.photonMap = (Photon*)photonMapBuffer.device;
				rtHitDatas[1][RayShadow].data.photonMapStartIdxs = (int*)photonMapStartIdxs.device;
				rtHitDatas[1][RayShadow].data.NOLT = (int*)NOLT.device;

				rtHitDataBufferLeft.copy(rtHitDatas[0], sizeof(rtHitDatas[0]));
				rtHitDataBufferRight.copy(rtHitDatas[1], sizeof(rtHitDatas[1]));

				// free memory
				delete[] photonData;
				delete[] photonMapData;
				delete[] tempPhotons;
			}
			void terminate()
			{
				cudaFree(paras.randState);
			}
		};
	}
}
namespace OpenGL
{
	namespace VR
	{
		struct RTRTGIVR : OpenGL
		{
			SourceManager sm;
			VRDevice hmd;
			OptiXTransCrossfire trans;
			OptiXVRRendererCrossfire renderer;
			CUDA::OptiX::PathTracing pathTracer;

			RTRTGIVR()
				:
				sm(),
				hmd(false),
				trans(&hmd, { 0.01,10 }),
				renderer(&sm, hmd.frameScale, &hmd),
				pathTracer(&sm, hmd.frameScale, trans.buffer.device)
			{
			}
			~RTRTGIVR()
			{
				pathTracer.terminate();
			}
			virtual void init(FrameScale const& _size)override
			{
				renderer.resize(_size);
				pathTracer.resize(hmd.frameScale, renderer.pixelBufferLeft.buffer, renderer.pixelBufferRight.buffer);
				renderer.use();
				glDisable(GL_MULTISAMPLE);
				//glEnable(GL_DEPTH_TEST);
			}
			virtual void run() override
			{
				trans.update();

				trans.operate(false);
				pathTracer.run();
				//trans.operate(true);
				//pathTracer.run();
				renderer.run();
			}
			virtual void frameSize(int _w, int _h)override
			{
				renderer.resize({ _w,_h });
			}
			virtual void framePos(int, int) override {}
			virtual void frameFocus(int) override {}
			virtual void mouseButton(int _button, int _action, int _mods)override {}
			virtual void mousePos(double _x, double _y)override {}
			virtual void mouseScroll(double _x, double _y)override {}
			virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
			{
				{
					switch (_key)
					{
					case GLFW_KEY_ESCAPE:if (_action == GLFW_PRESS)
						glfwSetWindowShouldClose(_window, true); break;
						//case GLFW_KEY_A:trans.key.refresh(0, _action); break;
						//case GLFW_KEY_D:trans.key.refresh(1, _action); break;
						//case GLFW_KEY_W:trans.key.refresh(2, _action); break;
						//case GLFW_KEY_S:trans.key.refresh(3, _action); break;
						/*	case GLFW_KEY_UP:monteCarlo.trans.persp.increaseV(0.02); break;
							case GLFW_KEY_DOWN:monteCarlo.trans.persp.increaseV(-0.02); break;
							case GLFW_KEY_RIGHT:monteCarlo.trans.persp.increaseD(0.01); break;
							case GLFW_KEY_LEFT:monteCarlo.trans.persp.increaseD(-0.01); break;*/
					}
				}
			}
		};
	}
}

int main()
{
	OpenGL::OpenGLInit(4, 5);
	Window::Window::Data winPara
	{
		"RTRTGIVR",
		{
			{1080 * 2,1200},
			true,false
		}
	};
	Window::WindowManager wm(winPara);
	OpenGL::VR::RTRTGIVR pathTracer;
	wm.init(0, &pathTracer);
	glfwSwapInterval(1);

	FPS fps;
	fps.refresh();
	while (!wm.close())
	{
		wm.pullEvents();
		wm.render();
		wm.swapBuffers();
		fps.refresh();
		//fps.printFPSAndFrameTime(2, 3);
		//wm.windows[0].data.setTitle(fps.str);
	}
	vr::VR_Shutdown();
	return 1;
}