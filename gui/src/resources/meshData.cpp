#include "./meshData.hpp"

MeshData::MeshData()
{
    vertexBuffer       = VK_NULL_HANDLE;
    vertexBufferMemory = VK_NULL_HANDLE;
    indexBuffer        = VK_NULL_HANDLE;
    indexBufferMemory  = VK_NULL_HANDLE;
    indexCount         = 0;
    modelMatrix        = glm::mat4(1.f);
}

void MeshData::clear(const VulkanContext& vkCtx)
{
    vkDeviceWaitIdle(vkCtx.device);

    if (vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vkCtx.device, vertexBuffer, nullptr);
        vkFreeMemory(vkCtx.device, vertexBufferMemory, nullptr);
        vertexBuffer       = VK_NULL_HANDLE;
        vertexBufferMemory = VK_NULL_HANDLE;
    }

    if (indexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vkCtx.device, indexBuffer, nullptr);
        vkFreeMemory(vkCtx.device, indexBufferMemory, nullptr);
        indexBuffer       = VK_NULL_HANDLE;
        indexBufferMemory = VK_NULL_HANDLE;
    }

    indexCount  = 0;
    modelMatrix = glm::mat4(1.f);
}
