#pragma once

#include "../common.hpp"
#include "../graphics/rendererUtils.hpp"

class MeshManager {
  public:
    struct VulkanContext {
        VkPhysicalDevice physicalDevice;
        VkDevice         device;
        VkCommandPool    commandPool;
        VkQueue          queue;
    };

  public:
    void init(VulkanContext vulkanContext);
    void cleanup();

    void loadMesh(std::string filePath);
    void bindMeshResources(VkCommandBuffer commandBuffer);

    void drawMesh(VkCommandBuffer commandBuffer);

  private:
    VulkanContext vulkanContext;

    struct {
        VkBuffer       vertexBuffer       = VK_NULL_HANDLE;
        VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
        VkBuffer       indexBuffer        = VK_NULL_HANDLE;
        VkDeviceMemory indexBufferMemory  = VK_NULL_HANDLE;

        uint32_t indexCount = 0;
    } currentMeshData;

  private:
    void cleanupCurrentMesh();

    void createVertexBuffer(const std::vector<Vertex>& vertices);
    void createIndexBuffer(const std::vector<uint32_t>& indices);
};
