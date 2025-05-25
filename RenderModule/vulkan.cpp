#include "stdafx.h"
#include "vulkan.h"

#include <iostream>
#include <bitset>
#include <span>
#include <tuple>

//#include "vulkan/vulkan_core.h"
#include "spirv-reflect/spirv_reflect.h"
#define GLFW_INCLUDE_VULKAN
#include <cstring>

#include "GLFW/glfw3.h"
#include "imgui/backends/imgui_impl_vulkan.h"
#include "imgui/backends/imgui_impl_glfw.h"

#include "shader_module.h"

namespace RS
{
#define RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(name) RS_ASSERT_COMPILE(sizeof(name) == sizeof(name##_T*), STRINGFY(name) "의 크기가 전방선언한 " STRING_APPEND(name, _T) "포인터와 크기가 다릅니다.")
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkBuffer);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkDeviceMemory);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkDescriptorSetLayout);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkPipelineLayout);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkPipeline);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkDescriptorSet);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkAccelerationStructureKHR);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkImage);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkImageView);
    RS_CHECK_FORWARD_DECLARATION_STRUCTURE_SIZE_FOR_VULKAN(VkDescriptorPool);

    RS_ASSERT_COMPILE(sizeof(DeviceAddress) == sizeof(VkDeviceAddress), "");

#ifdef NDEBUG
    const bool ON_DEBUG = false;
#else
    const bool ON_DEBUG = true;
#endif

#define RS_USE_RAYTRACING
    const uint32_t SHADER_GROUP_HANDLE_SIZE = 32;

    struct Global
    {
        uint32_t _width;
        uint32_t _height;

        VkInstance instance;
        VkDebugUtilsMessengerEXT debugMessenger;
        VkSurfaceKHR surface;

        VkPhysicalDevice physicalDevice;
        uint queueFamilyIndex;
        VkDevice device;

        VkRenderPass renderPass;

#if defined(RS_USE_RAYTRACING)
        PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
        PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
        PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
        PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
        PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
        PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
        PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
        PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
        PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;

        VkPhysicalDeviceRayTracingPipelinePropertiesKHR  rtProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
#endif

        const VkFormat swapChainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;    // intentionally chosen to match a specific format
        const VkColorSpaceKHR defaultSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        VkSwapchainKHR swapChain;
        vector<VkImage> swapChainImages;
        vector<VkImageView> swapChainImageViews;
        vector<VkFramebuffer> swapChainFrameBuffers;
        VkCommandPool commandPool;
        VkQueue graphicsQueue; // assume allowing graphics and present
        VkCommandBuffer commandBuffer;

        VkSemaphore imageAvailableSemaphore;
        VkSemaphore renderFinishedSemaphore;
        VkFence fence0;

        ~Global();

    }vk;

    inline Global::~Global() // Some methods works batter on this way.
    {
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (auto& item : vk.swapChainImageViews)
        {
            vkDestroyImageView(device, item, nullptr);
        }
        for (auto& item : vk.swapChainFrameBuffers)
        {
            vkDestroyFramebuffer(vk.device, item, nullptr);
        }

        vkDestroyFence(device, fence0, nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        vkDestroyDevice(device, nullptr);
        if (ON_DEBUG) {
            ((PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vk.instance, "vkDestroyDebugUtilsMessengerEXT"))
                (vk.instance, vk.debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

#pragma region Util
    VkDescriptorType convertDescriptorTypeToVulkan(const DescriptorType& type)
    {
#define CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(name) case DescriptorType::name: return VK_DESCRIPTOR_TYPE_##name
        switch (type)
        {
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(SAMPLER);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(COMBINED_IMAGE_SAMPLER);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(SAMPLED_IMAGE);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(STORAGE_IMAGE);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(UNIFORM_TEXEL_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(STORAGE_TEXEL_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(UNIFORM_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(STORAGE_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(UNIFORM_BUFFER_DYNAMIC);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(STORAGE_BUFFER_DYNAMIC);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(INPUT_ATTACHMENT);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(INLINE_UNIFORM_BLOCK);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(ACCELERATION_STRUCTURE_KHR);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(ACCELERATION_STRUCTURE_NV);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(SAMPLE_WEIGHT_IMAGE_QCOM);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(BLOCK_MATCH_IMAGE_QCOM);
            CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(MUTABLE_EXT);
            //CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(INLINE_UNIFORM_BLOCK_EXT);
            //CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT(MUTABLE_VALVE);
        default:
            return VK_DESCRIPTOR_TYPE_MAX_ENUM;
        }
#undef CONVERT_DESCRIPTORTYPE_TO_VULKAN_ELEMENT
    }
    DescriptorType convertDesctiptorTypeToEngine(const VkDescriptorType& type)
    {
#define CONVERT_DESCRIPTORTYPE_TO_ENGINE(name) case VK_DESCRIPTOR_TYPE_##name: return DescriptorType::name
        switch (type)
        {
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(SAMPLER);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(COMBINED_IMAGE_SAMPLER);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(SAMPLED_IMAGE);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(STORAGE_IMAGE);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(UNIFORM_TEXEL_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(STORAGE_TEXEL_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(UNIFORM_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(STORAGE_BUFFER);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(UNIFORM_BUFFER_DYNAMIC);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(STORAGE_BUFFER_DYNAMIC);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(INPUT_ATTACHMENT);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(INLINE_UNIFORM_BLOCK);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(ACCELERATION_STRUCTURE_KHR);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(ACCELERATION_STRUCTURE_NV);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(SAMPLE_WEIGHT_IMAGE_QCOM);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(BLOCK_MATCH_IMAGE_QCOM);
            CONVERT_DESCRIPTORTYPE_TO_ENGINE(MUTABLE_EXT);
            //CONVERT_DESCRIPTORTYPE_TO_ENGINE(INLINE_UNIFORM_BLOCK_EXT);
            //CONVERT_DESCRIPTORTYPE_TO_ENGINE(MUTABLE_VALVE);
        default:
            return DescriptorType::MAX_ENUM;
        }
#undef CONVERT_DESCRIPTORTYPE_TO_ENGINE
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
    {
        const char* severity;
        switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: severity = "[Verbose]"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: severity = "[Warning]"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: severity = "[Error]"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: severity = "[Info]"; break;
        default: severity = "[Unknown]";
        }

        const char* types;
        switch (messageType) {
        case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: types = "[General]"; break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: types = "[Performance]"; break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT: types = "[Validation]"; break;
        default: types = "[Unknown]";
        }

        std::cout << "[Debug]" << severity << types << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    bool checkValidationLayerSupport(vector<const char*>& reqestNames)
    {
        uint32_t count;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        vector<VkLayerProperties> availables(count);
        vkEnumerateInstanceLayerProperties(&count, availables.data());

        for (const char* reqestName : reqestNames) {
            bool found = false;

            for (const auto& available : availables) {
                if (strcmp(reqestName, available.layerName) == 0) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                return false;
            }
        }

        return true;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device, vector<const char*>& reqestNames)
    {
        uint32_t count;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        vector<VkExtensionProperties> availables(count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, availables.data());

        for (const char* reqestName : reqestNames) {
            bool found = false;

            for (const auto& available : availables) {
                if (strcmp(reqestName, available.extensionName) == 0) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                return false;
            }
        }

        return true;
    }

    uint findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags reqMemProps)
    {
        uint memTypeIndex = 0;
        std::bitset<32> isSuppoted(memoryTypeBits);

        VkPhysicalDeviceMemoryProperties spec;
        vkGetPhysicalDeviceMemoryProperties(vk.physicalDevice, &spec);

        for (auto& [props, _] : std::span<VkMemoryType>(spec.memoryTypes, spec.memoryTypeCount)) {
            if (isSuppoted[memTypeIndex] && (props & reqMemProps) == reqMemProps) {
                break;
            }
            ++memTypeIndex;
        }
        return memTypeIndex;
    }

    std::tuple<VkImage, VkDeviceMemory> createImage(VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags reqMemProps)
    {
        VkImage image;
        VkDeviceMemory imageMemory;

        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = { extent.width, extent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if (vkCreateImage(vk.device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(vk.device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, reqMemProps),
        };

        if (vkAllocateMemory(vk.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(vk.device, image, imageMemory, 0);

        return { image, imageMemory };
    }
    VkImageView createImageView(const VkImage image, const VkFormat& format, const VkImageSubresourceRange& subresourceRange)
    {
        VkImageViewCreateInfo ci0{
           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
           .image = image,
           .viewType = VK_IMAGE_VIEW_TYPE_2D,
           .format = format,
           .components = VkComponentMapping{ VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
           .subresourceRange = subresourceRange,
        };

        VkImageView imageView;
        vkCreateImageView(vk.device, &ci0, nullptr, &imageView);

        return imageView;
    }
    void setImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout oldImageLayout, VkImageLayout newImageLayout, VkImageSubresourceRange subresourceRange, 
        VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = oldImageLayout,
            .newLayout = newImageLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = subresourceRange,
        };

        if (oldImageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        else if (oldImageLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        }

        if (newImageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        else if (newImageLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        }

        vkCmdPipelineBarrier(
            cmdbuffer, srcStageMask, dstStageMask, 0,
            0, nullptr, 0, nullptr, 1, &barrier);
    }

    inline VkDeviceAddress getDeviceAddressOf(VkBuffer buffer)
    {
        VkBufferDeviceAddressInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = buffer,
        };
        return vk.vkGetBufferDeviceAddressKHR(vk.device, &info);
    }
    inline VkDeviceAddress getDeviceAddressOf(VkAccelerationStructureKHR as)
    {
        VkAccelerationStructureDeviceAddressInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = as,
        };
        return vk.vkGetAccelerationStructureDeviceAddressKHR(vk.device, &info);
    }
#pragma endregion

    void createVkInstance(GLFWwindow* window)
    {
        VkApplicationInfo appInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Hello Triangle",
            .apiVersion = VK_API_VERSION_1_3
        };

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (ON_DEBUG) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        vector<const char*> validationLayers;
        if (ON_DEBUG) validationLayers.push_back("VK_LAYER_KHRONOS_validation");
        if (!checkValidationLayerSupport(validationLayers)) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
        };

        VkInstanceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = ON_DEBUG ? &debugCreateInfo : nullptr,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = (uint)validationLayers.size(),
            .ppEnabledLayerNames = validationLayers.data(),
            .enabledExtensionCount = (uint)extensions.size(),
            .ppEnabledExtensionNames = extensions.data(),
        };

        if (vkCreateInstance(&createInfo, nullptr, &vk.instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        if (ON_DEBUG) {
            auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vk.instance, "vkCreateDebugUtilsMessengerEXT");
            if (!func || func(vk.instance, &debugCreateInfo, nullptr, &vk.debugMessenger) != VK_SUCCESS) {
                throw std::runtime_error("failed to set up debug messenger!");
            }
        }

        if (glfwCreateWindowSurface(vk.instance, window, nullptr, &vk.surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }
    
    void createVkDevice()
    {
        vk.physicalDevice = VK_NULL_HANDLE;

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(vk.instance, &deviceCount, nullptr);
        vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(vk.instance, &deviceCount, devices.data());

        vector<const char*> extentions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,

            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // not used
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // not used
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,

            VK_KHR_SPIRV_1_4_EXTENSION_NAME, // not used
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        };

        for (const auto& device : devices)
        {
            if (checkDeviceExtensionSupport(device, extentions))
            {
                vk.physicalDevice = device;
                break;
            }
        }

        if (vk.physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &queueFamilyCount, nullptr);
        vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &queueFamilyCount, queueFamilies.data());

        vk.queueFamilyIndex = 0;
        {
            for (; vk.queueFamilyIndex < queueFamilyCount; ++vk.queueFamilyIndex)
            {
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(vk.physicalDevice, vk.queueFamilyIndex, vk.surface, &presentSupport);

                if (queueFamilies[vk.queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT && presentSupport)
                    break;
            }

            if (vk.queueFamilyIndex >= queueFamilyCount)
                throw std::runtime_error("failed to find a graphics & present queue!");
        }
        float queuePriority = 1.0f;

        VkDeviceQueueCreateInfo queueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = vk.queueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        };

        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.shaderInt64 = TRUE;

        VkDeviceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = (uint)extentions.size(),
            .ppEnabledExtensionNames = extentions.data(),
            .pEnabledFeatures = &deviceFeatures, 
        };

        VkPhysicalDeviceBufferDeviceAddressFeatures f1{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            .bufferDeviceAddress = VK_TRUE,
        };

        VkPhysicalDeviceAccelerationStructureFeaturesKHR f2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            .accelerationStructure = VK_TRUE,
        };

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR f3{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            .rayTracingPipeline = VK_TRUE,
        };

        createInfo.pNext = &f1;
        f1.pNext = &f2;
        f2.pNext = &f3;

        if (vkCreateDevice(vk.physicalDevice, &createInfo, nullptr, &vk.device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(vk.device, vk.queueFamilyIndex, 0, &vk.graphicsQueue);
    }

    void loadDeviceExtensionFunctions(VkDevice device)
    {
        vk.vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)(vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
        vk.vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
        vk.vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
        vk.vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
        vk.vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
        vk.vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
        vk.vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)(vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
        vk.vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)(vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
        vk.vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)(vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));

        VkPhysicalDeviceProperties2 deviceProperties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &vk.rtProperties,
        };
        vkGetPhysicalDeviceProperties2(vk.physicalDevice, &deviceProperties2);

        if (vk.rtProperties.shaderGroupHandleSize != SHADER_GROUP_HANDLE_SIZE) {
            throw std::runtime_error("shaderGroupHandleSize must be 32 mentioned in the vulakn spec (Table 69. Required Limits)!");
        }
    }

    void createSwapChain()
    {
        VkSurfaceCapabilitiesKHR capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physicalDevice, vk.surface, &capabilities);

        {
            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physicalDevice, vk.surface, &formatCount, nullptr);
            vector<VkSurfaceFormatKHR> formats(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physicalDevice, vk.surface, &formatCount, formats.data());

            bool found = false;
            for (auto format : formats) {
                if (format.format == vk.swapChainImageFormat && format.colorSpace == vk.defaultSpace) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                throw std::runtime_error("");
            }
        }

        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        {
            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physicalDevice, vk.surface, &presentModeCount, nullptr);
            vector<VkPresentModeKHR> presentModes(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physicalDevice, vk.surface, &presentModeCount, presentModes.data());

            for (auto mode : presentModes) {
                if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                    break;
                }
            }
        }

        uint imageCount = capabilities.minImageCount + 1;
        VkSwapchainCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = vk.surface,
            .minImageCount = imageCount,
            .imageFormat = vk.swapChainImageFormat,
            .imageColorSpace = vk.defaultSpace,
            .imageExtent = {.width = vk._width , .height = vk._height },
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
        };

        if (vkCreateSwapchainKHR(vk.device, &createInfo, nullptr, &vk.swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(vk.device, vk.swapChain, &imageCount, nullptr);
        vk.swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(vk.device, vk.swapChain, &imageCount, vk.swapChainImages.data());

        {
            VkAttachmentDescription attachment = {};
            attachment.format = vk.swapChainImageFormat;
            attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;// wd->ClearEnable ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference color_attachment = {};
            color_attachment.attachment = 0;
            color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkSubpassDescription subpass = {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &color_attachment;
            VkSubpassDependency dependency = {};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            VkRenderPassCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            info.attachmentCount = 1;
            info.pAttachments = &attachment;
            info.subpassCount = 1;
            info.pSubpasses = &subpass;
            info.dependencyCount = 1;
            info.pDependencies = &dependency;
            vkCreateRenderPass(vk.device, &info, nullptr, &vk.renderPass);
        }

        // Create The Image Views
        vk.swapChainImageViews.resize(vk.swapChainImages.size());
        uint32 i = 0;
        for (const auto& image : vk.swapChainImages) {
            VkImageViewCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = vk.swapChainImageFormat,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1,
                },
            };

            vkCreateImageView(vk.device, &createInfo, nullptr, &vk.swapChainImageViews[i]);
            ++i;
        }

        {
            vk.swapChainFrameBuffers.resize(vk.swapChainImageViews.size());
            uint32 i = 0;
            VkImageView attachment[1];
            VkFramebufferCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = vk.renderPass;
            info.attachmentCount = 1;
            info.pAttachments = attachment;
            info.width = vk._width;
            info.height = vk._height;
            info.layers = 1;
            for (const auto& imageView : vk.swapChainImageViews)
            {
                attachment[0] = imageView;
                vkCreateFramebuffer(vk.device, &info, nullptr, &vk.swapChainFrameBuffers[i]);
                ++i;
            }
        }
    }

    void createCommandCenter()
    {
        VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = vk.queueFamilyIndex,
        };

        if (vkCreateCommandPool(vk.device, &poolInfo, nullptr, &vk.commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk.commandPool,
            .commandBufferCount = 1,
        };

        if (vkAllocateCommandBuffers(vk.device, &allocInfo, &vk.commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createSyncObjects()
    {
        VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        };
        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        if (vkCreateSemaphore(vk.device, &semaphoreInfo, nullptr, &vk.imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(vk.device, &semaphoreInfo, nullptr, &vk.renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(vk.device, &fenceInfo, nullptr, &vk.fence0) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }

    }

    IImage createImage()
    {
        VkImage image;
        VkDeviceMemory imageMemory;
        VkImageView imageView;

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM; //VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB(==vk.swapChainImageFormat)
        std::tie(image, imageMemory) = createImage(
            { vk._width, vk._height },
            format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkImageSubresourceRange subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        };
        imageView = createImageView(image, format, subresourceRange);

        vkResetCommandBuffer(vk.commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(vk.commandBuffer, &beginInfo);
        {
            setImageLayout(vk.commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
        }
        vkEndCommandBuffer(vk.commandBuffer);

        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &vk.commandBuffer,
        };
        vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk.graphicsQueue);

        return IImage(DescriptorType::STORAGE_IMAGE, image, imageMemory, imageView);
    }

    void initVulkan(const uint32_t width, const uint32_t height, GLFWwindow* window)
    {
        vk._width = width;
        vk._height = height;

        createVkInstance(window);
        createVkDevice();
        //glfwGetFramebufferSize(window, &vk._width, &vk._height);
        loadDeviceExtensionFunctions(vk.device);
        createSwapChain();
        createCommandCenter();
        createSyncObjects();

        // imgui
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        ImGui_ImplGlfw_InitForVulkan(window, true);
        ImGui_ImplVulkan_InitInfo init_info = {};
        //init_info.ApiVersion = VK_API_VERSION_1_3;              // Pass in your value of VkApplicationInfo::apiVersion, otherwise will default to header version.
        init_info.Instance = vk.instance;
        init_info.PhysicalDevice = vk.physicalDevice;
        init_info.Device = vk.device;
        init_info.QueueFamily = vk.queueFamilyIndex;
        init_info.Queue = vk.graphicsQueue;
        //init_info.PipelineCache = g_PipelineCache;
        //init_info.DescriptorPool = g_DescriptorPool;
        init_info.DescriptorPoolSize = 2;
        init_info.RenderPass = vk.renderPass;
        //init_info.UseDynamicRendering = true;
        //init_info.Subpass = 0;
        init_info.MinImageCount = vk.swapChainImages.size();
        init_info.ImageCount = vk.swapChainImages.size();
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        //init_info.Allocator = vk.allocator;
        //init_info.CheckVkResultFn = check_vk_result;
        ImGui_ImplVulkan_Init(&init_info);
    }

    void destroyVulkan()
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        ImGui_ImplVulkanH_DestroyWindow(vk.instance, vk.device, nullptr, nullptr);
    }

    IBuffer::~IBuffer()
    {
        //vkDestroyBuffer(vk.device, _buffer, nullptr);
        //vkFreeMemory(vk.device, _memory, nullptr);
    }
    IImage::~IImage()
    {
        //vkDestroyImageView(vk.device, _imageView, nullptr);
        //vkDestroyImage(vk.device, _image, nullptr);
        //vkFreeMemory(vk.device, _memory, nullptr);
    }
    AccelerationStructure::AccelerationStructure(IBuffer buffer, VkAccelerationStructureKHR_T* as) 
        : IResource(DescriptorType::ACCELERATION_STRUCTURE_KHR)
        , _buffer(buffer)
        , _accelerationStructure(as)
    {
        //blas인 경우에만 들어와야함
        _deviceAddress = getDeviceAddressOf(_accelerationStructure);
    }
    AccelerationStructure::AccelerationStructure(IBuffer buffer, VkAccelerationStructureKHR_T* as, vector<AccelerationStructure>&& blasArr)
        : IResource(DescriptorType::ACCELERATION_STRUCTURE_KHR)
        , _buffer(buffer)
        , _accelerationStructure(as)
        , _blasArr(std::move(blasArr))
    {
        //tlas인 경우에만 들어와야함
    }
    AccelerationStructure::~AccelerationStructure()
    {
        //vkDestroyBuffer(device, tlasBuffer, nullptr);
        //vkFreeMemory(device, tlasBufferMem, nullptr);
        //vkDestroyAccelerationStructureKHR(vk.device, _accelerationStructure, nullptr);
    }

    const bool IBuffer::upload(void* data, const uint32 size)
    {
        void* dst;
        vkMapMemory(vk.device, _memory, 0, size, 0, &dst);
        memcpy(dst, data, size);
        vkUnmapMemory(vk.device, _memory);

        return true;
    }

    const DeviceAddress IBuffer::getBufferAddress() const
    {
        return getDeviceAddressOf(_buffer);
    }

    IPipeline::Descriptor::~Descriptor()
    {
        //vkDestroyDescriptorPool(vk.device, _descriptorPool, nullptr);
        //vkDestroyDescriptorSetLayout(vk.device, _descriptorLayout, nullptr);
    }

    IPipeline::IPipeline()
    {
    }

    IPipeline::IPipeline(vector<IPipeline::Descriptor>&& descriptorArr, vector<VkDescriptorSet_T*>&& descriptorSetArr, VkPipelineLayout_T* pipelineLayout, VkPipeline_T* pipeline)
        : _descriptorArr(move(descriptorArr))
        , _descriptorSetArr(move(descriptorSetArr))
        , _pipelineLayout(pipelineLayout)
        , _pipeline(pipeline)
    {
    }

    IPipeline::~IPipeline()
    {
        //vkDestroyBuffer(vk.device, _sbtBuffer._buffer, nullptr);
        //vkFreeMemory(vk.device, _sbtBuffer._memory, nullptr);
        //
        //vkDestroyPipeline(vk.device, _pipeline, nullptr);
        //vkDestroyPipelineLayout(vk.device, _pipelineLayout, nullptr);
    }

    IPipeline::IPipeline(IPipeline&& rhs)
    {
        operator=(std::move(rhs));
    }

    IBuffer createBuffer(const void* data,
        const uint64       size,
        const DescriptorType descType,
        const uint32       usage,
        const uint32       memoryPropertyFlags,
        const char* debugName)
    {
        VkBuffer       buffer = {};
        VkDeviceMemory bufferMemory = {};

        // 1) Create the buffer
        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (vkCreateBuffer(vk.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }

#if defined(_RS_DEBUG_)
        // 2) Name it for the debugger
        PFN_vkSetDebugUtilsObjectNameEXT pfnSetDebugUtilsObjectNameEXT =
            (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
                vk.device, "vkSetDebugUtilsObjectNameEXT");
        VkDebugUtilsObjectNameInfoEXT debugInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_BUFFER,
            .objectHandle = (uint64_t)buffer,
            .pObjectName = debugName,
        };
        pfnSetDebugUtilsObjectNameEXT(vk.device, &debugInfo);
#endif

        // 3) Get memory requirements
        VkMemoryRequirements memRequirements = {};
        vkGetBufferMemoryRequirements(vk.device, buffer, &memRequirements);

        // 4) Prepare allocation info (keep flagsInfo alive)
        VkMemoryAllocateFlagsInfo flagsInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext = nullptr,
            .flags = static_cast<VkMemoryAllocateFlags>(
                (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
                    ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR
                    : 0u),
        };

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = (flagsInfo.flags != 0) ? &flagsInfo : nullptr,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(
                memRequirements.memoryTypeBits,
                memoryPropertyFlags),
        };
        if (vkAllocateMemory(vk.device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }

        // 5) Bind and (optionally) upload data
        vkBindBufferMemory(vk.device, buffer, bufferMemory, 0);

        if (data != nullptr) {
            uint8_t* dst = nullptr;
            vkMapMemory(vk.device, bufferMemory, 0, size, 0, (void**)&dst);
            memcpy(dst, data, size);
            vkUnmapMemory(vk.device, bufferMemory);
        }

        return IBuffer(descType, buffer, bufferMemory);
    }

    IPrimitiveBuffer createPrimitiveBuffer(const void* vertexData, const unsigned long long vertexBufferSize, const void* indexData, const unsigned long long indexBufferSize)
    {
        IBuffer vertexBuffer = createBuffer(
            vertexData,
            vertexBufferSize,
            DescriptorType::MAX_ENUM,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            "VertexBuffer");
        IBuffer indexBuffer = createBuffer(
            indexData,
            indexBufferSize,
            DescriptorType::MAX_ENUM,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            "IndexBuffer");

        return IPrimitiveBuffer(std::move(vertexBuffer), std::move(indexBuffer));
    }

    void createShaderBindingTable(const RaytracingPipelineDescription& info, IPipeline& pipeline)
    {
        const uint32 raygenShaderCount = info._raygenSource.size();
        const uint32 missShaderCount = info._missSource.size();
        const uint32 chitShaderCount = info._chitSource.size();
        const uint32 shaderCount = raygenShaderCount + missShaderCount + chitShaderCount;

        struct ShaderGroupHandle {
            uint8_t data[SHADER_GROUP_HANDLE_SIZE];
        };
        auto alignTo = [](auto value, auto alignment) -> decltype(value) {
            return (value + (decltype(value))alignment - 1) & ~((decltype(value))alignment - 1);
            };
        const uint32_t handleSize = SHADER_GROUP_HANDLE_SIZE;
        const uint32_t groupCount = shaderCount;
        vector<ShaderGroupHandle> handles(groupCount);
        vk.vkGetRayTracingShaderGroupHandlesKHR(vk.device, pipeline._pipeline, 0, groupCount, handleSize * groupCount, handles.data());

        vector<ShaderGroupHandle> raygenHandleArr;
        raygenHandleArr.reserve(raygenShaderCount);
        for (uint32 i = 0; i < raygenShaderCount; ++i)
            raygenHandleArr.push_back(ShaderGroupHandle{ handles[i] });
        vector<ShaderGroupHandle> missHandleArr;
        missHandleArr.reserve(missShaderCount);
        for (uint32 i = 0; i < missShaderCount; ++i)
            missHandleArr.push_back(ShaderGroupHandle{ handles[raygenShaderCount + i] });
        vector<ShaderGroupHandle> hitgHandleArr;
        hitgHandleArr.reserve(chitShaderCount);
        for (uint32 i = 0; i < chitShaderCount; ++i)
            hitgHandleArr.push_back(ShaderGroupHandle{ handles[raygenShaderCount + missShaderCount + i] });

        const uint32 raygenCustomDataStride = info._raygenShaderBindingTable._customDataStride;
        const uint32 missCustomDataStride = info._missShaderBindingTable._customDataStride;
        const uint32 chitCustomDataStride = info._chitShaderBindingTable._customDataStride;
        const uint32 raygenShaderBindingCount = info._raygenShaderBindingTable._sourceIndexArr.size();
        const uint32 missShaderBindingCount = info._missShaderBindingTable._sourceIndexArr.size();
        const uint32 chitShaderBindingCount = info._chitShaderBindingTable._sourceIndexArr.size();

        const uint32_t rgenStride = alignTo(handleSize + raygenCustomDataStride, vk.rtProperties.shaderGroupHandleAlignment);
        pipeline._rgenSbt.assign(rs_new VkStridedDeviceAddressRegionKHR{ 0, rgenStride, rgenStride * raygenShaderBindingCount });

        const uint64_t missOffset = alignTo(pipeline._rgenSbt->size, vk.rtProperties.shaderGroupBaseAlignment);
        const uint32_t missStride = alignTo(handleSize + missCustomDataStride, vk.rtProperties.shaderGroupHandleAlignment);
        pipeline._missSbt.assign(rs_new VkStridedDeviceAddressRegionKHR{ 0, missStride, missStride * missShaderBindingCount });

        const uint64_t hitgOffset = alignTo(missOffset + pipeline._missSbt->size, vk.rtProperties.shaderGroupBaseAlignment);
        const uint32_t hitgStride = alignTo(handleSize + chitCustomDataStride, vk.rtProperties.shaderGroupHandleAlignment);
        pipeline._hitgSbt.assign(rs_new VkStridedDeviceAddressRegionKHR{ 0, hitgStride, hitgStride * chitShaderBindingCount });

        const uint64_t sbtSize = hitgOffset + pipeline._hitgSbt->size;
        pipeline._sbtBuffer = std::move(createBuffer(
            nullptr,
            sbtSize,
            DescriptorType::MAX_ENUM,
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            "SBTBuffer"));

        auto sbtAddress = getDeviceAddressOf(pipeline._sbtBuffer._buffer);
        if (sbtAddress != alignTo(sbtAddress, vk.rtProperties.shaderGroupBaseAlignment)) {
            throw std::runtime_error("It will not be happened!");
        }
        pipeline._rgenSbt->deviceAddress = sbtAddress;
        pipeline._missSbt->deviceAddress = sbtAddress + missOffset;
        pipeline._hitgSbt->deviceAddress = sbtAddress + hitgOffset;

        uint8_t* dst;
        vkMapMemory(vk.device, pipeline._sbtBuffer._memory, 0, sbtSize, 0, (void**)&dst);
        {
            for (uint32 i = 0; i < raygenShaderBindingCount; ++i)
            {
                *(ShaderGroupHandle*)(dst + 0 + i * rgenStride) = raygenHandleArr[info._raygenShaderBindingTable._sourceIndexArr[i]];
                if (raygenCustomDataStride != 0)
                    std::memcpy(dst + 0 + i * rgenStride + handleSize, info._raygenShaderBindingTable._customDataArr[i], raygenCustomDataStride);
            }
            for (uint32 i = 0; i < missShaderBindingCount; ++i)
            {
                *(ShaderGroupHandle*)(dst + missOffset + i * missStride) = missHandleArr[info._missShaderBindingTable._sourceIndexArr[i]];
                if (missCustomDataStride != 0)
                    std::memcpy(dst + missOffset + i * missStride + handleSize, info._missShaderBindingTable._customDataArr[i], missCustomDataStride);
            }
            for (uint32 i = 0; i < chitShaderBindingCount; ++i)
            {
                *(ShaderGroupHandle*)(dst + hitgOffset + i * hitgStride) = hitgHandleArr[info._chitShaderBindingTable._sourceIndexArr[i]];
                if (chitCustomDataStride != 0)
                    std::memcpy(dst + hitgOffset + i * hitgStride + handleSize, info._chitShaderBindingTable._customDataArr[i], chitCustomDataStride);
            }
        }
        vkUnmapMemory(vk.device, pipeline._sbtBuffer._memory);
    }
    IPipeline createRayTracingPipelineInternal(const RaytracingPipelineDescription& info)
    {
        const uint32 raygenShaderCount = info._raygenSource.size();
        const uint32 missShaderCount = info._missSource.size();
        const uint32 chitShaderCount = info._chitSource.size();

        vector<ShaderModule<VK_SHADER_STAGE_RAYGEN_BIT_KHR>> raygenModuleArr;
        raygenModuleArr.reserve(raygenShaderCount);
        vector<ShaderModule<VK_SHADER_STAGE_MISS_BIT_KHR>> missModuleArr;
        missModuleArr.reserve(missShaderCount);
        vector<ShaderModule<VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>> chitModuleArr;
        chitModuleArr.reserve(chitShaderCount);

        vector<SpvReflectShaderModule> rayGenReflectModuleArr;
        rayGenReflectModuleArr.resize(raygenShaderCount);
        vector<SpvReflectShaderModule> missReflectModuleArr;
        missReflectModuleArr.resize(missShaderCount);
        vector<SpvReflectShaderModule> chitReflectModuleArr;
        chitReflectModuleArr.resize(chitShaderCount);

        uint32 descriptorSetMaxCount = 0;
        for (uint32 i = 0; i < raygenShaderCount; ++i)
        {
            const char* src = info._raygenSource[i];

            raygenModuleArr.push_back(std::move(ShaderModule<VK_SHADER_STAGE_RAYGEN_BIT_KHR>(vk.device, src)));

            ShaderModule<VK_SHADER_STAGE_RAYGEN_BIT_KHR>& shaderModule = raygenModuleArr[raygenModuleArr.size() - 1];
            SpvReflectShaderModule& reflectModule = rayGenReflectModuleArr[i];
            SpvReflectResult result = spvReflectCreateShaderModule(shaderModule._codeSize, shaderModule._code.data(), &reflectModule);
            RS_ASSERT_DEV("이영호", result == SPV_REFLECT_RESULT_SUCCESS, "Shader Relfeciton실패. 제보 부탁드립니다.");

            for(uint32 i = 0; i < reflectModule.descriptor_set_count; ++i)
                descriptorSetMaxCount = Math::getMax(descriptorSetMaxCount, reflectModule.descriptor_sets[i].set);
        }
        for (uint32 i = 0; i < missShaderCount; ++i)
        {
            const char* src = info._missSource[i];

            missModuleArr.push_back(std::move(ShaderModule<VK_SHADER_STAGE_MISS_BIT_KHR>(vk.device, src)));

            ShaderModule<VK_SHADER_STAGE_MISS_BIT_KHR>& raygenModule = missModuleArr[missModuleArr.size() - 1];
            SpvReflectShaderModule& reflectModule = missReflectModuleArr[i];
            SpvReflectResult result = spvReflectCreateShaderModule(raygenModule._codeSize, raygenModule._code.data(), &reflectModule);
            RS_ASSERT_DEV("이영호", result == SPV_REFLECT_RESULT_SUCCESS, "Shader Relfeciton실패. 제보 부탁드립니다.");

            for (uint32 i = 0; i < reflectModule.descriptor_set_count; ++i)
                descriptorSetMaxCount = Math::getMax(descriptorSetMaxCount, reflectModule.descriptor_sets[i].set);
        }
        for (uint32 i = 0; i < chitShaderCount; ++i)
        {
            const char* src = info._chitSource[i];

            chitModuleArr.push_back(std::move(ShaderModule<VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>(vk.device, src)));

            ShaderModule<VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>& shaderModule = chitModuleArr[chitModuleArr.size() - 1];
            SpvReflectShaderModule& reflectModule = chitReflectModuleArr[i];
            SpvReflectResult result = spvReflectCreateShaderModule(shaderModule._codeSize, shaderModule._code.data(), &reflectModule);
            RS_ASSERT_DEV("이영호", result == SPV_REFLECT_RESULT_SUCCESS, "Shader Relfeciton실패. 제보 부탁드립니다.");

            for (uint32 i = 0; i < reflectModule.descriptor_set_count; ++i)
                descriptorSetMaxCount = Math::getMax(descriptorSetMaxCount, reflectModule.descriptor_sets[i].set);
        }

        ++descriptorSetMaxCount;    // index를 count로 변환하기 위해서 +1

        vector<IPipeline::Descriptor> descriptorArr;
        descriptorArr.resize(descriptorSetMaxCount);

        auto makeResourceBindingTable = [](const VkShaderStageFlagBits stageFlag, const SpvReflectDescriptorSet& descriptorSet, vector<IPipeline::Descriptor>& descriptorArr)
            {
                const uint32 bindingCount = descriptorSet.binding_count;
                for (uint32 j = 0; j < bindingCount; ++j)
                {
                    const SpvReflectDescriptorBinding* binding = descriptorSet.bindings[j];
                    const char* name = binding->name;
                    if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    {
                        //RS_ASSERT_DEV("이영호", std::strcmp(name, "") == 0, "StorageBuffer를 사용하는 경우 TypeName(%s)을 비워주실 것을 권고합니다.", name);
                        RS_ASSERT_DEV("이영호", binding->type_description->member_count == 1, "StorageBuffer를 사용하는 경우 Member는 하나여야합니다. Count: %d", binding->type_description->member_count);
                        RS_ASSERT_DEV("이영호", binding->type_description->members->op == SpvOpTypeRuntimeArray, "StorageBuffer를 사용하는 경우 Member는 배열이어야합니다.");
                        if(std::strcmp(name, "") == 0)
                            name = binding->type_description->members->struct_member_name;
                    }
                    else
                    {
                        RS_ASSERT_DEV("이영호", std::strcmp(name, "") != 0, "Shader에 Binding된 Resource의 Name이 비어있습니다.");
                    }
                    const uint32 setIndex = binding->set;
                    const SpvReflectDescriptorType descType = binding->descriptor_type;
                    const uint32 bindingIndex = binding->binding;

                    uint32 findIndex = 0xffffffff;
                    IPipeline::Descriptor& descriptor = descriptorArr[setIndex];
                    const uint32 resourceBindingCount = descriptor._resourceBindingTable.size();
                    for (uint32 k = 0; k < resourceBindingCount; ++k)
                    {
                        IPipeline::Descriptor::ResourceBinding& resourceBinding = descriptor._resourceBindingTable[k];
                        if (std::strcmp(name, resourceBinding._name) == 0)
                        {
                            findIndex = k;
                            break;
                        }
                    }

                    auto convertReflectDescriptorTypeToDescriptorType = [](const SpvReflectDescriptorType descType) -> VkDescriptorType
                        {
                            switch (descType)
                            {
                            case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
                                return VK_DESCRIPTOR_TYPE_SAMPLER;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                                return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                                return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                                return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                                return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                                return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                                return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                            case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                                return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                            default:
                            {
                                RS_ASSERT_DEV("이영호", false, "지원하지 않는 Shader DescriptorType입니다. 제보해주세요.");
                                return VK_DESCRIPTOR_TYPE_MAX_ENUM;
                            }
                            }
                        };
                    if (findIndex != 0xffffffff)
                    {
                        RS_ASSERT_DEV("이영호", descriptor._resourceBindingTable[findIndex]._binding == bindingIndex, "두 개의 Pipeline에서 같은 DescriptorSet에 대하여 같은 Resource에 다른 BindingIndex 검출.");
                        RS_ASSERT_DEV("이영호", descriptor._resourceBindingTable[findIndex]._descriptorType == convertDesctiptorTypeToEngine(convertReflectDescriptorTypeToDescriptorType(descType)), "두 개의 Pipeline에서 같은 DescriptorSet에 대하여 같은 Resource Name에 다른 DescriptorType 검출.");
                        descriptor._resourceBindingTable[findIndex]._stageFlags |= stageFlag;
                    }
                    else
                    {
                        descriptor._resourceBindingTable.push_back(
                            IPipeline::Descriptor::ResourceBinding{ 
                                name, 
                                bindingIndex, 
                                convertDesctiptorTypeToEngine(convertReflectDescriptorTypeToDescriptorType(descType)),
                                (uint32)stageFlag
                            });
                    }
                }
            };

        for (const SpvReflectShaderModule& reflectModule : rayGenReflectModuleArr)
        {
            for (uint32 i = 0; i < reflectModule.descriptor_set_count; ++i)
            {
                const SpvReflectDescriptorSet& descriptorSet = reflectModule.descriptor_sets[i];
                makeResourceBindingTable(VK_SHADER_STAGE_RAYGEN_BIT_KHR, descriptorSet, descriptorArr);
            }
        }
        for (const SpvReflectShaderModule& reflectModule : missReflectModuleArr)
        {
            for (uint32 i = 0; i < reflectModule.descriptor_set_count; ++i)
            {
                const SpvReflectDescriptorSet& descriptorSet = reflectModule.descriptor_sets[i];
                makeResourceBindingTable(VK_SHADER_STAGE_MISS_BIT_KHR, descriptorSet, descriptorArr);
            }
        }
        for (const SpvReflectShaderModule& reflectModule : chitReflectModuleArr)
        {
            for (uint32 i = 0; i < reflectModule.descriptor_set_count; ++i)
            {
                const SpvReflectDescriptorSet& descriptorSet = reflectModule.descriptor_sets[i];
                makeResourceBindingTable(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, descriptorSet, descriptorArr);
            }
        }

        vector<VkDescriptorSet> descriptorSetArr;
        descriptorSetArr.resize(descriptorSetMaxCount);
        for (uint32 i = 0; i < descriptorSetMaxCount; ++i)
        {
            vector<VkDescriptorPoolSize> poolSizes;
            poolSizes.reserve(descriptorArr[i]._resourceBindingTable.size());
            vector<VkDescriptorSetLayoutBinding> bindings;
            bindings.reserve(descriptorArr[i]._resourceBindingTable.size());
            for(IPipeline::Descriptor::ResourceBinding& binding : descriptorArr[i]._resourceBindingTable)
            {
                VkDescriptorType descType = convertDescriptorTypeToVulkan(binding._descriptorType);
                bindings.push_back(VkDescriptorSetLayoutBinding{
                        .binding = binding._binding,
                        .descriptorType = descType,
                        .descriptorCount = 1,
                        .stageFlags = binding._stageFlags,
                    });

                poolSizes.push_back({ descType, 1 });
            };

            VkDescriptorSetLayoutCreateInfo layoutInfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = static_cast<uint32>(bindings.size()),
                .pBindings = bindings.data(),
            };
            vkCreateDescriptorSetLayout(vk.device, &layoutInfo, nullptr, &descriptorArr[i]._descriptorLayout);

            VkDescriptorPoolCreateInfo poolInfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1,
                .poolSizeCount = static_cast<uint32>(poolSizes.size()),
                .pPoolSizes = poolSizes.data(),
            };
            vkCreateDescriptorPool(vk.device, &poolInfo, nullptr, &descriptorArr[i]._descriptorPool);

            VkDescriptorSetAllocateInfo ai0{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptorArr[i]._descriptorPool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptorArr[i]._descriptorLayout,
            };
            vkAllocateDescriptorSets(vk.device, &ai0, &descriptorSetArr[i]);
        }

        vector<VkDescriptorSetLayout> descriptorLayoutArr;
        descriptorLayoutArr.reserve(descriptorArr.size());
        for (const IPipeline::Descriptor& desc : descriptorArr)
            descriptorLayoutArr.push_back(desc._descriptorLayout);

        VkPipelineLayout pipelineLayout;
        VkPipelineLayoutCreateInfo ci1{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = static_cast<uint32>(descriptorLayoutArr.size()),
            .pSetLayouts = descriptorLayoutArr.data(),
        };
        vkCreatePipelineLayout(vk.device, &ci1, nullptr, &pipelineLayout);
        
        vector<VkPipelineShaderStageCreateInfo> stages;
        stages.reserve(raygenShaderCount + missShaderCount + chitShaderCount);
        for (ShaderModule<VK_SHADER_STAGE_RAYGEN_BIT_KHR>& shaderModule : raygenModuleArr)
            stages.push_back(shaderModule);
        for (ShaderModule<VK_SHADER_STAGE_MISS_BIT_KHR>& shaderModule : missModuleArr)
            stages.push_back(shaderModule);
        for (ShaderModule<VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>& shaderModule : chitModuleArr)
            stages.push_back(shaderModule);

        uint32 shaderIndex = 0;
        vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
        shaderGroups.reserve(raygenShaderCount + missShaderCount + chitShaderCount);
        for (uint32 i = 0; i < raygenShaderCount; ++i)
        {
            shaderGroups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader = shaderIndex++,
                .closestHitShader = VK_SHADER_UNUSED_KHR,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
                });
        }
        for (uint32 i = 0; i < missShaderCount; ++i)
        {
            shaderGroups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader = shaderIndex++,
                .closestHitShader = VK_SHADER_UNUSED_KHR,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
                });
        }
        for (uint32 i = 0; i < chitShaderCount; ++i)
        {
            shaderGroups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                .generalShader = VK_SHADER_UNUSED_KHR,
                .closestHitShader = shaderIndex++,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
                });
        }

        VkPipeline pipeline;
        VkRayTracingPipelineCreateInfoKHR ci2{
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            .stageCount = static_cast<uint32>(stages.size()),
            .pStages = stages.data(),
            .groupCount = static_cast<uint32>(shaderGroups.size()),
            .pGroups = shaderGroups.data(),
            .maxPipelineRayRecursionDepth = info._maxPipelineRayRecursionDepth,
            .layout = pipelineLayout,
        };
        vk.vkCreateRayTracingPipelinesKHR(vk.device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci2, nullptr, &pipeline);

        return IPipeline(move(descriptorArr), move(descriptorSetArr), pipelineLayout, pipeline);
    }
    IPipeline createRayTracingPipeline(const RaytracingPipelineDescription& info)
    {
        IPipeline pipeline = createRayTracingPipelineInternal(info);
        createShaderBindingTable(info, pipeline);

        return std::move(pipeline);
    }

    struct WriteDescriptorSet
    {
        VkWriteDescriptorSet _write;
        VkDescriptorImageInfo _imageInfo;
        VkDescriptorBufferInfo _bufferInfo;
        VkWriteDescriptorSetAccelerationStructureKHR _asInfo;
    };
    Ptr<WriteDescriptorSet> createWriteDescriptorSet(VkDescriptorSet descriptorSet, const uint32 dstBinding, IResource* resource)
    {
        VkWriteDescriptorSet write_temp{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .descriptorCount = 1,
        };

        VkDescriptorType descType = convertDescriptorTypeToVulkan(resource->_descriptorType);

        switch (descType)
        {
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        {
            WriteDescriptorSet* ttt = rs_new WriteDescriptorSet{ write_temp, {}, {}, {} };
            ttt->_asInfo = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &(static_cast<AccelerationStructure*>(resource)->_accelerationStructure),
            };
            ttt->_write.pNext = &ttt->_asInfo;
            ttt->_write.dstBinding = dstBinding;
            ttt->_write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            return ttt;
        }
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        {
            WriteDescriptorSet* ttt = rs_new WriteDescriptorSet{ write_temp, {}, {}, {} };
            ttt->_imageInfo = {
                .imageView = static_cast<IImage*>(resource)->_imageView,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            ttt->_write.dstBinding = dstBinding;
            ttt->_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ttt->_write.pImageInfo = &ttt->_imageInfo;
            return ttt;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        {
            WriteDescriptorSet* ttt = rs_new WriteDescriptorSet{ write_temp, {}, {}, {} };
            ttt->_bufferInfo = {
                .buffer = static_cast<IBuffer*>(resource)->_buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE,
            };
            ttt->_write.dstBinding = dstBinding;
            ttt->_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ttt->_write.pBufferInfo = &ttt->_bufferInfo;
            return ttt;
        }
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        {
            WriteDescriptorSet* ttt = rs_new WriteDescriptorSet{ write_temp, {}, {}, {} };
            ttt->_bufferInfo = {
                .buffer = static_cast<IBuffer*>(resource)->_buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE,
            };
            ttt->_write.dstBinding = dstBinding;
            ttt->_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ttt->_write.pBufferInfo = &ttt->_bufferInfo;
            return ttt;
        }
        default:
        {
            RS_ASSERT_DEV("이영호", false, "올바르지 않거나 지원되지 않는 DescriptorType. 지원이 필요하면 요청 부탁드립니다.");
            return rs_new WriteDescriptorSet{ write_temp, {}, {}, {} };
        }
        }
    }

    static uint32 imageIndex = uint32_max;  // #TODO- 반드시 나중에 RenderPass로 옮겨야함
    static const IImage* gOutImage = nullptr;
    void startFrame()
    {
        static const VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

        vkWaitForFences(vk.device, 1, &vk.fence0, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &vk.fence0);

        vkAcquireNextImageKHR(vk.device, vk.swapChain, UINT64_MAX, vk.imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        vkResetCommandBuffer(vk.commandBuffer, 0);
        if (vkBeginCommandBuffer(vk.commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
    }
    void endFrame()
    {
        if (vkEndCommandBuffer(vk.commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        VkSemaphore waitSemaphores[] = {
            vk.imageAvailableSemaphore
        };
        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };

        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = sizeof(waitSemaphores) / sizeof(waitSemaphores[0]),
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &vk.commandBuffer,
        };

        VkResult result = vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, vk.fence0);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .swapchainCount = 1,
            .pSwapchains = &vk.swapChain,
            .pImageIndices = &imageIndex,
        };

        vkQueuePresentKHR(vk.graphicsQueue, &presentInfo);

        vkQueueWaitIdle(vk.graphicsQueue);
    }

    void bindRenderPass()
    {
        VkClearValue        ClearValue;
        ClearValue.color.float32[0] = 0.45f;
        ClearValue.color.float32[1] = 0.55f;
        ClearValue.color.float32[2] = 0.60f;
        ClearValue.color.float32[3] = 1.00f;
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = vk.renderPass;
        info.framebuffer = vk.swapChainFrameBuffers[imageIndex];
        info.renderArea.extent.width = vk._width;
        info.renderArea.extent.height = vk._height;
        //info.clearValueCount = 1;
        //info.pClearValues = &ClearValue;
        vkCmdBeginRenderPass(vk.commandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
    void unbindRenderPass()
    {
        ImGui::Render();

        // Record dear imgui primitives into command buffer
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vk.commandBuffer);

        // Submit command buffer
        vkCmdEndRenderPass(vk.commandBuffer);
    }

    static const IPipeline* gCurrentPipeline = nullptr;

    const bool bindBufferInternal(const uint32 setIndex, const uint32 bindIndex, IResource* resource)
    {
        RS_ASSERT_DEV("이영호", gCurrentPipeline != nullptr, "Pipeline이 먼저 bind되어야합니다.");
        RS_ASSERT_DEV("이영호", resource != nullptr, "Bind하고자 하는 Resource가 nullptr입니다.");
        
        Ptr<WriteDescriptorSet> write = createWriteDescriptorSet(gCurrentPipeline->_descriptorSetArr[setIndex], bindIndex, resource);
        vkUpdateDescriptorSets(vk.device, 1, &write->_write, 0, VK_NULL_HANDLE);
        return true;
    }
    const bool bindPipelineInternal(const IPipeline& pipeline, const IImage& outImage)
    {
        RS_ASSERT_DEV("이영호", gCurrentPipeline == nullptr, "이미 BindPipeline을 했다면, unBindPipeline전에는 bindPipeline이 호출되면 안됩니다.");
        gCurrentPipeline = &pipeline;
        gOutImage = &outImage;

        vkCmdBindPipeline(vk.commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, gCurrentPipeline->_pipeline);

        return true;
    }
    const bool unbindPipelineInternal()
    {
        RS_ASSERT_DEV("이영호", gCurrentPipeline != nullptr, "bindPipeline이 먼저 호출되어야합니다.");

        static const VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        static const VkImageCopy copyRegion = {
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .extent = { vk._width, vk._height, 1 },
        };

        setImageLayout(
            vk.commandBuffer,
            gOutImage->_image,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            subresourceRange);

        setImageLayout(
            vk.commandBuffer,
            vk.swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange);

        vkCmdCopyImage(
            vk.commandBuffer,
            gOutImage->_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.swapChainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion);

        setImageLayout(
            vk.commandBuffer,
            gOutImage->_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            subresourceRange);

        setImageLayout(
            vk.commandBuffer,
            vk.swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            subresourceRange);

        gCurrentPipeline = nullptr;

        return true;
    }

    void traceRays()
    {
        static const VkStridedDeviceAddressRegionKHR callSbt{};

        const vector<VkDescriptorSet>& descSets = gCurrentPipeline->_descriptorSetArr;
        vkCmdBindDescriptorSets(
            vk.commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            gCurrentPipeline->_pipelineLayout, 0, static_cast<uint32_t>(descSets.size()), descSets.data(), 0, 0);

        vk.vkCmdTraceRaysKHR(
            vk.commandBuffer,
            gCurrentPipeline->_rgenSbt.get(),
            gCurrentPipeline->_missSbt.get(),
            gCurrentPipeline->_hitgSbt.get(),
            &callSbt,
            vk._width, vk._height, 1);
    }

    VkTransformMatrixKHR convertMatrixToVkTransformMatrix(const float4x4& matrix)
    {
        VkTransformMatrixKHR vkMatrix;
        vkMatrix.matrix[0][0] = matrix._rows[0][0];
        vkMatrix.matrix[0][1] = matrix._rows[0][1];
        vkMatrix.matrix[0][2] = matrix._rows[0][2];

        vkMatrix.matrix[1][0] = matrix._rows[1][0];
        vkMatrix.matrix[1][1] = matrix._rows[1][1];
        vkMatrix.matrix[1][2] = matrix._rows[1][2];

        vkMatrix.matrix[2][0] = matrix._rows[2][0];
        vkMatrix.matrix[2][1] = matrix._rows[2][1];
        vkMatrix.matrix[2][2] = matrix._rows[2][2];

        vkMatrix.matrix[0][3] = matrix._rows[3][0];
        vkMatrix.matrix[1][3] = matrix._rows[3][1];
        vkMatrix.matrix[2][3] = matrix._rows[3][2];

        return vkMatrix;
    };
    AccelerationStructure createBLAS(const vector<CreateBlasData>& blasData)
    {
        // #TODO- 나중에 Blas Key로 찾을 수 있는 Container를 제공해야할듯.

        const uint32 geometryCount = blasData.size();
        vector<IBuffer> vertexBufferArr;
        vertexBufferArr.reserve(geometryCount);
        vector<IBuffer> indexBufferArr;
        indexBufferArr.reserve(geometryCount);
        vector<IBuffer> geometryBufferArr;
        geometryBufferArr.reserve(geometryCount);

        vector<uint32> triangleCountArr;
        triangleCountArr.reserve(geometryCount);
        vector<VkAccelerationStructureGeometryKHR> asGeometoryArr;
        asGeometoryArr.reserve(geometryCount);
        vector<VkAccelerationStructureBuildRangeInfoKHR> buildBlasRangeInfo;
        buildBlasRangeInfo.reserve(geometryCount);
        for (uint32 i = 0; i < geometryCount; ++i)
        {
            const CreateBlasData& data = blasData[i];

            IBuffer vertexBuffer = createBuffer(
                data._vertexData, data._vertexStride * data._vertexCount, DescriptorType::MAX_ENUM,
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                "Blas_VertexBuffer");

            IBuffer indexBuffer = createBuffer(
                data._indexData, data._indexStride * data._indexCount, DescriptorType::MAX_ENUM,
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                "Blas_IndexBuffer");

            VkTransformMatrixKHR transform = convertMatrixToVkTransformMatrix(data._transform);
            IBuffer geometryBuffer = createBuffer(
                &transform, sizeof(transform), DescriptorType::MAX_ENUM, 
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                "Blas_TransformBuffer");

            VkAccelerationStructureGeometryKHR geometry{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry = {
                .triangles = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                    .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                    .vertexData = {.deviceAddress = getDeviceAddressOf(vertexBuffer._buffer) },
                    .vertexStride = data._vertexStride,
                    .maxVertex = data._vertexCount - 1,
                    .indexType = VK_INDEX_TYPE_UINT32,
                    .indexData = {.deviceAddress = getDeviceAddressOf(indexBuffer._buffer) },
                    .transformData = {.deviceAddress = getDeviceAddressOf(geometryBuffer._buffer) },
                },
            },
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
            };

            asGeometoryArr.push_back(std::move(geometry));
            const uint32 triangleCount = data._indexCount / 3;
            RS_ASSERT_DEV("이영호", (data._indexCount % 3) == 0, "Index Count는 반드시 3의 배수여아합니다. Triangle 구성을 위해");
            triangleCountArr.push_back(triangleCount);
            buildBlasRangeInfo.push_back({
                    .primitiveCount = triangleCount,
                    .transformOffset = 0,
                });

            vertexBufferArr.push_back(vertexBuffer);
            indexBufferArr.push_back(indexBuffer);
            geometryBufferArr.push_back(geometryBuffer);
        }

        VkAccelerationStructureBuildGeometryInfoKHR buildBlasInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .geometryCount = static_cast<uint32>(asGeometoryArr.size()),
            .pGeometries = asGeometoryArr.data(),
        };

        VkAccelerationStructureBuildSizesInfoKHR requiredSize{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        vk.vkGetAccelerationStructureBuildSizesKHR(
            vk.device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildBlasInfo,
            triangleCountArr.data(),
            &requiredSize);

        IBuffer blasBuffer = createBuffer(
            nullptr, requiredSize.accelerationStructureSize, DescriptorType::ACCELERATION_STRUCTURE_KHR, 
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            "BlasBuffer");

        IBuffer scratchBuffer = createBuffer(
            nullptr, requiredSize.buildScratchSize, DescriptorType::STORAGE_BUFFER, 
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            "Blas_ScratchBuffer");

        VkAccelerationStructureKHR blas;
        // Generate BLAS handle
        {
            VkAccelerationStructureCreateInfoKHR asCreateInfo{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                .buffer = blasBuffer._buffer,
                .size = requiredSize.accelerationStructureSize,
                .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            };
            vk.vkCreateAccelerationStructureKHR(vk.device, &asCreateInfo, nullptr, &blas);
        }

        // Build BLAS using GPU operations
        {
            vkResetCommandBuffer(vk.commandBuffer, 0);
            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            vkBeginCommandBuffer(vk.commandBuffer, &beginInfo);
            {
                buildBlasInfo.dstAccelerationStructure = blas;
                buildBlasInfo.scratchData.deviceAddress = getDeviceAddressOf(scratchBuffer._buffer);

                VkAccelerationStructureBuildGeometryInfoKHR buildBlasInfos[] = { buildBlasInfo };
                VkAccelerationStructureBuildRangeInfoKHR* buildBlasRangeInfos[] = { buildBlasRangeInfo.data() };
                vk.vkCmdBuildAccelerationStructuresKHR(vk.commandBuffer, 1, buildBlasInfos, buildBlasRangeInfos);
            }
            vkEndCommandBuffer(vk.commandBuffer);

            VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &vk.commandBuffer,
            };
            vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(vk.graphicsQueue);
        }

        return AccelerationStructure(std::move(blasBuffer), blas);
    }
    AccelerationStructure createTLAS(const vector<CreateTLASData>& dataArr)
    {
        // #TODO- 나중에 Tlas Key로 찾을 수 있는 Container를 제공해야할듯.

        const uint32 instanceCount = dataArr.size();

        vector<VkAccelerationStructureInstanceKHR> instanceArr;
        instanceArr.reserve(instanceCount);
        vector<AccelerationStructure> blasArr;

        for (uint32 i = 0; i < instanceCount; ++i)
        {
            const CreateTLASData& data = dataArr[i];
            AccelerationStructure blas = createBLAS(data._blasData);

            VkAccelerationStructureInstanceKHR instance{
                .instanceCustomIndex = data._instanceCustomIndex,
                .mask = 0xFF,
                .instanceShaderBindingTableRecordOffset = data._instanceShaderBindingTableRecordOffset,
                .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                .accelerationStructureReference = blas._deviceAddress,
            };
            instance.transform = convertMatrixToVkTransformMatrix(data._transform);

            instanceArr.push_back(std::move(instance));
            blasArr.push_back(std::move(blas));
        }

        IBuffer instanceBuffer = createBuffer(
            instanceArr.data(), sizeof(instanceArr[0]) * instanceArr.size(), DescriptorType::MAX_ENUM,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            "Tlas_InstanceBuffer");

        VkAccelerationStructureGeometryKHR instances{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry = {
                .instances = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                    .data = {.deviceAddress = getDeviceAddressOf(instanceBuffer._buffer) },
                },
            },
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        };

        VkAccelerationStructureBuildGeometryInfoKHR buildTlasInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .geometryCount = 1,     // It must be 1 with .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR as shown in the vulkan spec.
            .pGeometries = &instances,
        };

        VkAccelerationStructureBuildSizesInfoKHR requiredSize{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        vk.vkGetAccelerationStructureBuildSizesKHR(
            vk.device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildTlasInfo,
            &instanceCount,
            &requiredSize);

        IBuffer tlasBuffer = createBuffer(
            nullptr, requiredSize.accelerationStructureSize,DescriptorType::ACCELERATION_STRUCTURE_KHR, 
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            "TlasBuffer");

        IBuffer scratchBuffer = createBuffer(
            nullptr, requiredSize.buildScratchSize, DescriptorType::STORAGE_BUFFER, 
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            "Tlas_ScatchBuffer");

        VkAccelerationStructureKHR tlas;
        // Generate TLAS handle
        {
            VkAccelerationStructureCreateInfoKHR asCreateInfo{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                .buffer = tlasBuffer._buffer,
                .size = requiredSize.accelerationStructureSize,
                .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            };
            vk.vkCreateAccelerationStructureKHR(vk.device, &asCreateInfo, nullptr, &tlas);
        }

        // Build TLAS using GPU operations
        {
            vkResetCommandBuffer(vk.commandBuffer, 0);
            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            vkBeginCommandBuffer(vk.commandBuffer, &beginInfo);
            {
                buildTlasInfo.dstAccelerationStructure = tlas;
                buildTlasInfo.scratchData.deviceAddress = getDeviceAddressOf(scratchBuffer._buffer);

                VkAccelerationStructureBuildRangeInfoKHR buildTlasRangeInfo = { .primitiveCount = instanceCount };
                VkAccelerationStructureBuildRangeInfoKHR* buildTlasRangeInfo_[] = { &buildTlasRangeInfo };
                vk.vkCmdBuildAccelerationStructuresKHR(vk.commandBuffer, 1, &buildTlasInfo, buildTlasRangeInfo_);
            }
            vkEndCommandBuffer(vk.commandBuffer);

            VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &vk.commandBuffer,
            };
            vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(vk.graphicsQueue);
        }

        return AccelerationStructure(std::move(tlasBuffer), tlas, std::move(blasArr));
    }
}