#pragma once

#include <vulkan/vulkan_core.h>
#include <vulkan/vk_enum_string_helper.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <gmsh.h>

#include <cstdint>
#include <optional>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>

#define VK_CHECK(x)                                                                                \
    do {                                                                                           \
        VkResult result = (x);                                                                     \
        if (result != VK_SUCCESS) {                                                                \
            throw std::runtime_error(#x " failed with " + std::string(string_VkResult(result)));   \
        }                                                                                          \
    } while (0)

// ╭─────────────────────────────────────────────────────────╮
// │                    Helper Structures                    │
// ╰─────────────────────────────────────────────────────────╯

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete();
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;

    bool isAdequate();
};

struct VulkanContext {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkSurfaceKHR     surface        = VK_NULL_HANDLE;

    PFN_vkCmdSetPolygonModeEXT vkCmdSetPolygonModeEXT = nullptr;

    QueueFamilyIndices queueFamilies;
    VkQueue            graphicsQueue = VK_NULL_HANDLE;
    VkQueue            presentQueue  = VK_NULL_HANDLE;

    SwapchainSupportDetails swapchainSupport;
    int                     swapchainImageCount  = 0;
    VkFormat                swapchainImageFormat = VK_FORMAT_UNDEFINED;

    VkCommandPool graphicsCommandPool = VK_NULL_HANDLE;
};

struct AttachmentFormats {
    VkFormat color = VK_FORMAT_UNDEFINED;
    VkFormat depth = VK_FORMAT_UNDEFINED;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;

    static VkVertexInputBindingDescription                  getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

struct alignas(16) PushConstantData { // std430
    glm::mat4 mvpMatrix;
};

// ╭─────────────────────────────────────────────────────────╮
// │                     Helper Funtions                     │
// ╰─────────────────────────────────────────────────────────╯
namespace utils {
void createImage(VkPhysicalDevice      physicalDevice,
                 VkDevice              device,
                 uint32_t              width,
                 uint32_t              height,
                 VkFormat              format,
                 VkImageTiling         tiling,
                 VkImageUsageFlags     usage,
                 VkMemoryPropertyFlags properties,
                 VkImage*              image,
                 VkDeviceMemory*       imageMemory);

void createImageView(VkDevice           device,
                     VkImage            image,
                     VkFormat           format,
                     VkImageAspectFlags aspectFlags,
                     VkImageView*       imageView);

void createBuffer(VkDeviceSize          size,
                  VkBufferUsageFlags    usage,
                  VkMemoryPropertyFlags properties,
                  VkBuffer*             buffer,
                  VkDeviceMemory*       bufferMemory,
                  VkPhysicalDevice      physicalDevice,
                  VkDevice              device);

void copyBuffer(VkBuffer      srcBuffer,
                VkBuffer      dstBuffer,
                VkDeviceSize  size,
                VkDevice      device,
                VkCommandPool commandPool,
                VkQueue       queue);

uint32_t findMemoryType(uint32_t              typeFilter,
                        VkMemoryPropertyFlags properties,
                        VkPhysicalDevice      physicalDevice);
} // namespace utils
