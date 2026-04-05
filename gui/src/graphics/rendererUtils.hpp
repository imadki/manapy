#pragma once

#include "../common.hpp"

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

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;

    static VkVertexInputBindingDescription                  getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

struct alignas(16) PushConstantData { // std430
    glm::mat4 viewProjMat;
};

// ╭─────────────────────────────────────────────────────────╮
// │                     Helper Funtions                     │
// ╰─────────────────────────────────────────────────────────╯
namespace utils {
void createBuffer(VkDeviceSize          size,
                  VkBufferUsageFlags    usage,
                  VkMemoryPropertyFlags properties,
                  VkBuffer&             buffer,
                  VkDeviceMemory&       bufferMemory,
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
