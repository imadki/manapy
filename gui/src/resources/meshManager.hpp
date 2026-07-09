#pragma once

#include "../graphics/rendererUtils.hpp"
#include "./meshData.hpp"
#include "../ui/uiState.hpp"

class MeshManager {
  public:
  public:
    void init(const VulkanContext& vkCtx);
    void shutdown(const VulkanContext& vkCtx);

    void update(const VulkanContext& vkCtx, const UIState& uiState);

    const MeshData& getMeshData() const;

  private:
    MeshData meshData;

  private:
    void      loadMesh(const VulkanContext& vkCtx, const std::filesystem::path& filePath);
    glm::mat4 buildMeshModelMatrix(const std::vector<Vertex>& vertices);

    void createVertexBuffer(const VulkanContext&       vkCtx,
                            const std::vector<Vertex>& vertices,
                            VkBuffer*                  vertexBuffer,
                            VkDeviceMemory*            vertexBufferMemory);
    void createIndexBuffer(const VulkanContext&         vkCtx,
                           const std::vector<uint32_t>& indices,
                           VkBuffer*                    indexBuffer,
                           VkDeviceMemory*              indexBufferMemory);
};
