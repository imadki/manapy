#pragma once

#include <vulkan/vulkan_core.h>

#include "../rendererUtils.hpp"
#include "GLFW/glfw3.h"
#include "imgui_impl_vulkan.h"
#include "meshViewResources.hpp"
#include <unordered_map>
#include "../../ui/uiState.hpp"
#include "../../resources/meshData.hpp"

enum RenderPassEnum {
    UI,
    MESH_VIEW,
};

class RenderPassManager {

  public:
    void init(const VulkanContext& vkCtx, GLFWwindow* glfwWindow);
    void shutdown(const VulkanContext& vkCtx);

    void initMeshViewTextureDesc();
    void clearMeshViewTextureDesc();
    void update(const VulkanContext& vkCtx, const UIState& uiState);

    VkRenderPass                  getRenderPass(RenderPassEnum type);
    ImGui_ImplVulkan_PipelineInfo getImGuiPipelineInfo() const;

    VkDescriptorSet getMeshViewTextureDesc(uint32_t idx) const;

    void recordDrawCommandBuffer(const VulkanContext& vkCtx,
                                 VkCommandBuffer      commandBuffer,
                                 VkFramebuffer        swapchainFramebuffer,
                                 VkExtent2D           swapchainExtent,
                                 uint32_t             frameIdx,
                                 uint32_t             imageIdx,
                                 PushConstantData     pushConstant,
                                 const UIState&       uiState,
                                 const MeshData&      meshData);

  private:
    MeshViewResources meshViewResources;

    std::unordered_map<RenderPassEnum, VkRenderPass> renderPasses;

  private:
    void createRenderPasses(const VulkanContext& vkCtx, AttachmentFormats meshViewFormats);
};
