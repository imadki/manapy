#pragma once

#include "../rendererUtils.hpp"

class MeshPipeline {
  public:
    void init(const VulkanContext& vkCtx, VkRenderPass renderPass);
    void shutdown(const VulkanContext& vkCtx);

    VkPipeline       getGraphicsPipeline() const;
    VkPipelineLayout getGraphicsPipelineLayout() const;

  private:
    VkPipeline       pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  private:
    VkShaderModule createShaderModule(const VulkanContext& vkCtx, const char* bytecodePath);
};
