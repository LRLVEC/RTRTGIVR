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

namespace CUDA
{
	namespace OptiX
	{
		struct PathTracing :RayTracer
		{
			Context context;
			OptixModuleCompileOptions moduleCompileOptions;
			OptixPipelineCompileOptions pipelineCompileOptions;
			ModuleManager mm;
			OptixProgramGroupOptions programGroupOptions;
			Program rayAllocator;
			Program miss;
			Program closestHit;
			unsigned int depthMax;
			OptixPipelineLinkOptions pipelineLinkOptions;
			Pipeline pip;
			SbtRecord<RayData> raygenData;
			SbtRecord<int> missData;
			SbtRecord<CloseHitData> hitData;
			Buffer raygenDataBuffer;
			Buffer missDataBuffer;
			Buffer hitDataBuffer;
			OptixShaderBindingTable sbt;
			Buffer frameBuffer;
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
			cudaArray* cuArray;
			cudaTextureObject_t tex;
			cudaResourceDesc texRes;
			cudaTextureDesc texDescr;
			CubeMap sky;

			PathTracing(OpenGL::SourceManager* _sourceManager, OpenGL::OptiXRenderer* dr, OpenGL::FrameScale const& _size, void* transInfoDevice)
				:
				context(),
				moduleCompileOptions{
				OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
				OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
				OPTIX_COMPILE_DEBUG_LEVEL_NONE },
				pipelineCompileOptions{ false,
				OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
				8/*payload nums*/,2,OPTIX_EXCEPTION_FLAG_NONE,"paras" },
				mm(&_sourceManager->folder, context, &moduleCompileOptions, &pipelineCompileOptions),
				programGroupOptions{},
				rayAllocator(Vector<String<char>>("__raygen__RayAllocator"), Program::RayGen, &programGroupOptions, context, &mm),
				miss(Vector<String<char>>("__miss__Ahh"), Program::Miss, &programGroupOptions, context, &mm),
				closestHit(Vector<String<char>>("__closesthit__Ahh"), Program::HitGroup, &programGroupOptions, context, &mm),
				depthMax(3),
				pipelineLinkOptions{ depthMax,OPTIX_COMPILE_DEBUG_LEVEL_NONE,false },
				pip(context, &pipelineCompileOptions, &pipelineLinkOptions, { rayAllocator ,closestHit, miss }),
				raygenDataBuffer(raygenData, false),
				missDataBuffer(missData, false),
				hitDataBuffer(hitData, false),
				sbt(),
				frameBuffer(*dr),
				parasBuffer(paras, false),
				box(_sourceManager->folder.find("resources/box.stl").readSTL()),
				vertices(Buffer::Device),
				normals(Buffer::Device),
				triangleBuildInput({}),
				accelOptions({}),
				GASOutput(Buffer::Device),
				sky("resources/skybox/")
			{
				box.getVerticesRepeated();
				box.getNormals();
				box.printInfo(false);

				cudaChannelFormatDesc channelDesc{ 8, 8, 8, 8, cudaChannelFormatKindUnsigned };
				cudaExtent extent{ sky.width, sky.width, 6 };
				cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayCubemap);
				sky.moveToGPU(cuArray);
				//cudaMemcpyToArray(cuArray, 0, 0, bmpdata, bmpsize, cudaMemcpyHostToDevice);//only for dim<3
				memset(&texRes, 0, sizeof(cudaResourceDesc));
				memset(&texDescr, 0, sizeof(cudaTextureDesc));
				texRes.resType = cudaResourceTypeArray;
				texRes.res.array.array = cuArray;
				texDescr.normalizedCoords = true;
				texDescr.filterMode = cudaFilterModePoint;
				texDescr.addressMode[0] = cudaAddressModeWrap;
				texDescr.addressMode[1] = cudaAddressModeWrap;
				texDescr.addressMode[2] = cudaAddressModeWrap;
				texDescr.readMode = cudaReadModeNormalizedFloat;
				cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);
				paras.cubeTexture = tex;

				vertices.copy(box.verticesRepeated.data, sizeof(Math::vec3<float>)* box.verticesRepeated.length);
				normals.copy(box.normals.data, sizeof(Math::vec3<float>)* box.normals.length);

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

				srand(time(nullptr));
				cudaMalloc(&paras.randState, _size.w* _size.h * sizeof(curandState));
				initRandom(paras.randState, rand(), 1024, (_size.w* _size.h + 1023) / 1024, _size.w* _size.h);

				paras.depthMax = depthMax;

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
				optixSbtRecordPackHeader(rayAllocator, &raygenData);
				raygenData.data = { 0.462f, 0.725f, 0.f };
				raygenDataBuffer.copy(raygenData);
				optixSbtRecordPackHeader(miss, &missData);
				missDataBuffer.copy(missData);
				optixSbtRecordPackHeader(closestHit, &hitData);
				hitData.data.normals = (float3*)normals.device;
				hitDataBuffer.copy(hitData);

				sbt.raygenRecord = raygenDataBuffer;
				sbt.missRecordBase = missDataBuffer;
				sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				sbt.missRecordCount = 1;
				sbt.hitgroupRecordBase = hitDataBuffer;
				sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<CloseHitData>);
				sbt.hitgroupRecordCount = 1;

				cudaStreamCreate(&cuStream);
			}
			virtual void run()
			{
				frameBuffer.map();
				optixLaunch(pip, cuStream, parasBuffer, sizeof(Parameters), &sbt, paras.size.x, paras.size.y, 1);
				cudaDeviceSynchronize();
				frameBuffer.unmap();
			}
			virtual void resize(OpenGL::FrameScale const& _size, GLuint _gl)
			{
				frameBuffer.resize(_gl);
				frameBuffer.map();
				paras.image = (float4*)frameBuffer.device;
				paras.size = make_uint2(_size.w, _size.h);
				parasBuffer.copy(paras);
				frameBuffer.unmap();
			}
			void terminate()
			{
				cudaDestroyTextureObject(tex);
				cudaFreeArray(cuArray);
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
			OptiXTrans trans;
			OptiXVRRenderer renderer;
			CUDA::OptiX::PathTracing pathTracer;

			RTRTGIVR()
				:
				sm(),
				hmd(false),
				trans(&hmd, { 0.01,10 }),
				renderer(&sm, hmd.frameScale, &hmd),
				pathTracer(&sm, &renderer, hmd.frameScale, trans.buffer.device)
			{
			}
			~RTRTGIVR()
			{
				pathTracer.terminate();
			}
			virtual void init(FrameScale const& _size)override
			{
				renderer.resize(_size);
				pathTracer.resize(hmd.frameScale, renderer);
				renderer.use();
				glDisable(GL_MULTISAMPLE);
				//glEnable(GL_DEPTH_TEST);
			}
			virtual void run() override
			{
				trans.update();

				trans.operate(false);
				pathTracer.run();
				renderer.refreshFrameData();
				renderer.renderLeft();

				trans.operate(true);
				pathTracer.run();
				renderer.refreshFrameData();
				renderer.renderRight();

				//renderer.renderWindow();

				renderer.commit();
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