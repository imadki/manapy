#pragma once

#include "../rendererUtils.hpp"
#include "./meshPipeline.hpp"
#include <vulkan/vulkan_core.h>

class MeshViewResources {
  public:
    AttachmentFormats resolveFormats(const VulkanContext& vkCtx);

    void init(const VulkanContext& vkCtx,
              VkRenderPass         renderPass,
              VkExtent2D           viewportExtent = {600, 400});

    void initTextureDesc();
    void clearTextureDesc();
    void shutdown(const VulkanContext& vkCtx);

    void reset(const VulkanContext& vkCtx, VkRenderPass renderPass, VkExtent2D viewportExtent);

    VkDescriptorSet getMeshViewTextureDesc(uint32_t idx) const;

    VkExtent2D       getImageExtent() const;
    VkFramebuffer    getFrameBuffer(uint32_t idx) const;
    VkPipeline       getGraphicsPipeline() const;
    VkPipelineLayout getGraphicsPipelineLayout() const;

  private:
    MeshPipeline meshPipeline;

    VkSampler sampler = VK_NULL_HANDLE;

    AttachmentFormats formats;
    VkExtent2D        extent;

    std::vector<VkImage>        colorImages;
    std::vector<VkDeviceMemory> colorImagesMemory;
    std::vector<VkImageView>    colorImageViews;

    VkImage        depthImage       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView    depthImageView   = VK_NULL_HANDLE;

    std::vector<VkFramebuffer> framebuffers;

    std::vector<VkDescriptorSet> descriptorSets;

  private:
    VkFormat findSupportedFormat(const VulkanContext&         vkCtx,
                                 const std::vector<VkFormat>& candidates,
                                 VkImageTiling                tiling,
                                 VkFormatFeatureFlags         features);

    VkFormat findDepthFormat(const VulkanContext& vkCtx);
};
