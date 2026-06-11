#pragma once

#include "../graphics/rendererUtils.hpp"
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>

struct MeshData {
    VkBuffer       vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer       indexBuffer;
    VkDeviceMemory indexBufferMemory;

    uint32_t indexCount;

    glm::mat4 modelMatrix;

    MeshData();
    void clear(const VulkanContext& vkCtx);
};
