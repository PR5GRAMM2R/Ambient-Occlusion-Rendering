#pragma once

#include <cstring>

#include "imgui/imgui.h"

struct GLFWwindow;
struct VkBuffer_T;
struct VkDeviceMemory_T;
struct VkDescriptorSetLayout_T;
struct VkPipelineLayout_T;
struct VkPipeline_T;
struct VkDescriptorSet_T;
struct VkAccelerationStructureKHR_T;
struct VkStridedDeviceAddressRegionKHR;
struct VkImage_T;
struct VkImageView_T;
struct VkAccelerationStructureKHR_t;
struct VkDescriptorPool_T;

namespace RS
{
	void initVulkan(const unsigned int width, const unsigned int height, GLFWwindow* window);
	void destroyVulkan();

	using DeviceAddress = uint64_t;

	enum class ShaderStageFlagBits
	{
		VERTEX_BIT, 
		TESSELLATION_CONTROL_BIT, 
		TESSELLATION_EVALUATION_BIT, 
		GEOMETRY_BIT, 
		FRAGMENT_BIT, 
		COMPUTE_BIT, 
		ALL_GRAPHICS, 
		ALL, 
		RAYGEN_BIT_KHR, 
		ANY_HIT_BIT_KHR, 
		CLOSEST_HIT_BIT_KHR, 
		MISS_BIT_KHR, 
		INTERSECTION_BIT_KHR, 
		CALLABLE_BIT_KHR, 
		TASK_BIT_EXT, 
		MESH_BIT_EXT, 
		SUBPASS_SHADING_BIT_HUAWEI, 
		CLUSTER_CULLING_BIT_HUAWEI, 
		RAYGEN_BIT_NV, 
		ANY_HIT_BIT_NV, 
		CLOSEST_HIT_BIT_NV, 
		MISS_BIT_NV, 
		INTERSECTION_BIT_NV, 
		CALLABLE_BIT_NV, 
		TASK_BIT_NV, 
		MESH_BIT_NV, 
		COUNT,
	};

	enum class DescriptorType
	{
		SAMPLER = 0,
		COMBINED_IMAGE_SAMPLER = 1,
		SAMPLED_IMAGE = 2,
		STORAGE_IMAGE = 3,
		UNIFORM_TEXEL_BUFFER = 4,
		STORAGE_TEXEL_BUFFER = 5,
		UNIFORM_BUFFER = 6,
		STORAGE_BUFFER = 7,
		UNIFORM_BUFFER_DYNAMIC = 8,
		STORAGE_BUFFER_DYNAMIC = 9,
		INPUT_ATTACHMENT = 10,
		INLINE_UNIFORM_BLOCK = 1000138000,
		ACCELERATION_STRUCTURE_KHR = 1000150000,
		ACCELERATION_STRUCTURE_NV = 1000165000,
		SAMPLE_WEIGHT_IMAGE_QCOM = 1000440000,
		BLOCK_MATCH_IMAGE_QCOM = 1000440001,
		MUTABLE_EXT = 1000351000,
		INLINE_UNIFORM_BLOCK_EXT = INLINE_UNIFORM_BLOCK,
		MUTABLE_VALVE = MUTABLE_EXT,
		MAX_ENUM = 0x7FFFFFFF
	};

	enum class BufferUsageFlagBits {
		BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001,
		BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002,
		BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
		BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
		BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010,
		BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020,
		BUFFER_USAGE_INDEX_BUFFER_BIT = 0x00000040,
		BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x00000080,
		BUFFER_USAGE_INDIRECT_BUFFER_BIT = 0x00000100,
		BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
		BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR = 0x00002000,
		BUFFER_USAGE_VIDEO_DECODE_DST_BIT_KHR = 0x00004000,
		BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT = 0x00000800,
		BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT = 0x00001000,
		BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT = 0x00000200,
#ifdef VK_ENABLE_BETA_EXTENSIONS
		BUFFER_USAGE_EXECUTION_GRAPH_SCRATCH_BIT_AMDX = 0x02000000,
#endif
		BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR = 0x00080000,
		BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR = 0x00100000,
		BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR = 0x00000400,
		BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR = 0x00008000,
		BUFFER_USAGE_VIDEO_ENCODE_SRC_BIT_KHR = 0x00010000,
		BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT = 0x00200000,
		BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT = 0x00400000,
		BUFFER_USAGE_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT = 0x04000000,
		BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT = 0x00800000,
		BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT = 0x01000000,
		BUFFER_USAGE_RAY_TRACING_BIT_NV = BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
		BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT = BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR = BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		BUFFER_USAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
	};

	enum class  MemoryPropertyFlagBits{
		MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001,
		MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002,
		MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004,
		MEMORY_PROPERTY_HOST_CACHED_BIT = 0x00000008,
		MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT = 0x00000010,
		MEMORY_PROPERTY_PROTECTED_BIT = 0x00000020,
		MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD = 0x00000040,
		MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD = 0x00000080,
		MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV = 0x00000100,
		MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
	};

	struct IResource
	{
		DescriptorType _descriptorType = DescriptorType::MAX_ENUM;

		IResource()
		{}
		IResource(const DescriptorType descriptorType)
			: _descriptorType(descriptorType)
		{}

		virtual ~IResource() {};
	};
	struct IBuffer final : public IResource
	{
		VkBuffer_T* _buffer = nullptr;
		VkDeviceMemory_T* _memory = nullptr;

		IBuffer()
			: _buffer(nullptr)
			, _memory(nullptr)
		{}
		IBuffer(const DescriptorType descType, VkBuffer_T* buffer, VkDeviceMemory_T* memory)
			: IResource(descType)
			, _buffer(buffer)
			, _memory(memory)
		{}
		virtual ~IBuffer() override final;

		// #TODO- ���߿� Offset, �� Size �ڵ����� ������ �ʿ��ϸ� ��������
		const bool upload(void* data, const uint32 size);

		const DeviceAddress getBufferAddress() const;
	};
	struct IImage final : public IResource
	{
		IImage()
		{}
		IImage(const DescriptorType descType, VkImage_T* image, VkDeviceMemory_T* memory, VkImageView_T* imageView)
			: IResource(descType)
			, _image(image)
			, _memory(memory)
			, _imageView(imageView)
		{}
		virtual ~IImage() override final;

		VkImage_T* _image = nullptr;
		VkDeviceMemory_T* _memory = nullptr;
		VkImageView_T* _imageView = nullptr;
	};
	struct AccelerationStructure final : public IResource
	{
		AccelerationStructure() {}
		AccelerationStructure(IBuffer buffer, VkAccelerationStructureKHR_T* as);
		AccelerationStructure(IBuffer buffer, VkAccelerationStructureKHR_T* as, vector<AccelerationStructure>&& blasArr);
		virtual ~AccelerationStructure() override final;

		IBuffer _buffer;
		VkAccelerationStructureKHR_T* _accelerationStructure = nullptr;
		DeviceAddress _deviceAddress = 0;
		vector<AccelerationStructure> _blasArr;
	};

	struct IPrimitiveBuffer
	{
		IBuffer _vertexbuffer;
		IBuffer _indexBuffer;

		IPrimitiveBuffer()
		{}
		IPrimitiveBuffer(IBuffer vertexbuffer, IBuffer indexBuffer)
			: _vertexbuffer(vertexbuffer)
			, _indexBuffer(indexBuffer)
		{}

		IPrimitiveBuffer(const IPrimitiveBuffer& rhs)
			: _vertexbuffer(rhs._vertexbuffer)
			, _indexBuffer(rhs._indexBuffer)
		{}

		void operator=(const IPrimitiveBuffer& rhs)
		{
			this->_vertexbuffer = rhs._vertexbuffer;
			this->_indexBuffer = rhs._indexBuffer;
		}
	};

	IPrimitiveBuffer createPrimitiveBuffer(const void* vertexData, const unsigned long long vertexBufferSize, const void* indexData, const unsigned long long indexBufferSize);

	struct IPipeline
	{
		struct Descriptor
		{
			struct ResourceBinding
			{
				const char* _name;
				uint32 _binding = 0xffffffff;
				DescriptorType _descriptorType = DescriptorType::MAX_ENUM;
				uint32 _stageFlags = 0;
			};

			~Descriptor();

			vector<ResourceBinding> _resourceBindingTable;
			VkDescriptorPool_T* _descriptorPool = nullptr;
			VkDescriptorSetLayout_T* _descriptorLayout = nullptr;
		};

		vector<Descriptor> _descriptorArr;
		vector<VkDescriptorSet_T*> _descriptorSetArr;
		VkPipelineLayout_T* _pipelineLayout = nullptr;
		VkPipeline_T* _pipeline = nullptr;

		IBuffer _sbtBuffer;
		Ptr<VkStridedDeviceAddressRegionKHR> _rgenSbt;
		Ptr<VkStridedDeviceAddressRegionKHR> _missSbt;
		Ptr<VkStridedDeviceAddressRegionKHR> _hitgSbt;

		IPipeline();
		IPipeline(vector<IPipeline::Descriptor>&& descriptorArr, vector<VkDescriptorSet_T*>&& descriptorSetArr, VkPipelineLayout_T* pipelineLayout, VkPipeline_T* pipeline);
		~IPipeline();

		IPipeline(IPipeline&& rhs);
		void operator=(IPipeline&& rhs)
		{
			this->_descriptorArr = move(rhs._descriptorArr);
			rhs._descriptorArr.clear();
			this->_descriptorSetArr = move(rhs._descriptorSetArr);
			rhs._descriptorSetArr.clear();
			this->_pipelineLayout = rhs._pipelineLayout;
			rhs._pipelineLayout = nullptr;
			this->_pipeline = rhs._pipeline;
			rhs._pipeline = nullptr;

			_sbtBuffer = std::move(rhs._sbtBuffer);

			_rgenSbt.swap(rhs._rgenSbt);
			_missSbt.swap(rhs._missSbt);
			_hitgSbt.swap(rhs._hitgSbt);
		}

	public:
		pair<uint32, uint32> findBindIndex(const char* name) const
		{
			const uint32 desciptorCount = _descriptorArr.size();
			for (uint32 i = 0; i < desciptorCount; ++i)
			{
				const uint32 resourceBindingCount = _descriptorArr[i]._resourceBindingTable.size();
				for (uint32 j = 0; j < resourceBindingCount; ++j)
				{
					const Descriptor::ResourceBinding& resourceBinding = _descriptorArr[i]._resourceBindingTable[j];
					if (std::strcmp(resourceBinding._name, name) == 0)	// strcmp �ؾ��Ҽ���..?
					{
						return pair<uint32, uint32>(i, resourceBinding._binding);
					}
				}
			}

			return pair<uint32, uint32>(0xffffffff, 0xffffffff);
		}
	};

	struct RaytracingPipelineDescription
	{
		vector<const char*> _raygenSource;
		vector<const char*> _missSource;
		vector<const char*> _chitSource;

		uint32 _maxPipelineRayRecursionDepth;

		struct CustomDataInfo
		{
			vector<uint32> _sourceIndexArr;
			uint32 _customDataStride = 0;
			vector<const void*> _customDataArr;
		};

		// first : source index, second : customdata
		CustomDataInfo _raygenShaderBindingTable;
		CustomDataInfo _missShaderBindingTable;
		CustomDataInfo _chitShaderBindingTable;
	};
	IPipeline createRayTracingPipeline(const RaytracingPipelineDescription& info);

	// for IMGUI
	void startFrame();
	void endFrame();

	void bindRenderPass();
	void unbindRenderPass();

	const bool bindBufferInternal(const uint32 setIndex, const uint32 bindIndex, IResource* resource);
#define bindBuffer(name, resource) \
{ \
	RS_ASSERT_DEV("�̿�ȣ", resource != nullptr, "Bind�ϰ��� �ϴ� Resource�� nullptr�Դϴ�."); \
	static pair<uint32, uint32> location = currentPipeline.findBindIndex(name); \
	bindBufferInternal(location._first, location._second, resource); \
}

	const bool bindPipelineInternal(const IPipeline& pipeline, const IImage& outImage);
#define bindPipeline(pipeline, image) \
{ \
	const IPipeline& currentPipeline = pipeline; \
	bindPipelineInternal(pipeline, image);
	

#define unbindPipeline() \
	unbindPipelineInternal(); \
}
	const bool unbindPipelineInternal();

	void traceRays();

	IBuffer createBuffer(const void* data, const uint64 size, const DescriptorType descType, const uint32 usage, const uint32 memoryPropertyFlags, const char* debugName);
	IImage createImage();
	struct CreateBlasData
	{
		const void* _vertexData;
		const uint32 _vertexStride;
		const uint32 _vertexCount;
		const void* _indexData;
		const uint32 _indexStride;
		const uint32 _indexCount;
		const float4x4 _transform;
	};
	struct CreateTLASData
	{
		const vector<CreateBlasData> _blasData;
		const uint32 _instanceCustomIndex;
		const uint32 _instanceShaderBindingTableRecordOffset;
		const float4x4 _transform;
	};
	AccelerationStructure createTLAS(const vector<CreateTLASData>& dataArr);
}