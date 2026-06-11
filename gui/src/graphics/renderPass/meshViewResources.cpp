#include "meshViewResources.hpp"
#include "../../common/config.hpp"

#include <cstdio>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan_core.h>

AttachmentFormats MeshViewResources::resolveFormats(const VulkanContext& vkCtx)
{
    if (formats.color == VK_FORMAT_UNDEFINED || formats.depth == VK_FORMAT_UNDEFINED) {
        formats.color = VK_FORMAT_B8G8R8A8_UNORM;
        formats.depth = findDepthFormat(vkCtx);
    }

    return AttachmentFormats{
        .color = formats.color,
        .depth = formats.depth,
    };
}

void MeshViewResources::init(const VulkanContext& vkCtx,
                             VkRenderPass         renderPass,
                             VkExtent2D           viewportExtent)
{
    resolveFormats(vkCtx);
    this->extent = viewportExtent;

    meshPipeline.init(vkCtx, renderPass);

    // ─[ Sampler ]────────────────────────────────────────────────────────
    VkSamplerCreateInfo samplerInfo{
        .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter               = VK_FILTER_LINEAR,
        .minFilter               = VK_FILTER_LINEAR,
        .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias              = 0.0f,
        .anisotropyEnable        = VK_FALSE,
        .compareEnable           = VK_FALSE,
        .compareOp               = VK_COMPARE_OP_ALWAYS,
        .minLod                  = 0.0f,
        .maxLod                  = 0.0f,
        .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VK_CHECK(vkCreateSampler(vkCtx.device, &samplerInfo, nullptr, &sampler));

    // ─[ Color Images ]───────────────────────────────────────────────────
    colorImages.resize(Config::renderer.maxFramesInFlight);
    colorImagesMemory.resize(Config::renderer.maxFramesInFlight);
    colorImageViews.resize(Config::renderer.maxFramesInFlight);

    for (size_t i = 0; i < Config::renderer.maxFramesInFlight; i++) {
        utils::createImage(vkCtx.physicalDevice,
                           vkCtx.device,
                           viewportExtent.width,
                           viewportExtent.height,
                           formats.color,
                           VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           &colorImages[i],
                           &colorImagesMemory[i]);

        utils::createImageView(vkCtx.device,
                               colorImages[i],
                               formats.color,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               &colorImageViews[i]);
    }

    // ─[ Depth Image ]────────────────────────────────────────────────────
    utils::createImage(vkCtx.physicalDevice,
                       vkCtx.device,
                       viewportExtent.width,
                       viewportExtent.height,
                       formats.depth,
                       VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &depthImage,
                       &depthImageMemory);

    utils::createImageView(vkCtx.device,
                           depthImage,
                           formats.depth,
                           VK_IMAGE_ASPECT_DEPTH_BIT,
                           &depthImageView);

    // ─[ Framebuffer ]────────────────────────────────────────────────────
    framebuffers.resize(Config::renderer.maxFramesInFlight);

    for (size_t i = 0; i < Config::renderer.maxFramesInFlight; i++) {
        std::array<VkImageView, 2> attachments = {colorImageViews[i], depthImageView};

        VkFramebufferCreateInfo framebufferInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = renderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .width           = viewportExtent.width,
            .height          = viewportExtent.height,
            .layers          = 1,
        };

        VK_CHECK(vkCreateFramebuffer(vkCtx.device, &framebufferInfo, nullptr, &framebuffers[i]));
    }
}

void MeshViewResources::initTextureDesc()
{
    descriptorSets.resize(Config::renderer.maxFramesInFlight);

    for (size_t i = 0; i < Config::renderer.maxFramesInFlight; i++) {
        descriptorSets[i] = ImGui_ImplVulkan_AddTexture(sampler,
                                                        colorImageViews[i],
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void MeshViewResources::clearTextureDesc()
{
    for (size_t i = 0; i < Config::renderer.maxFramesInFlight; i++) {
        ImGui_ImplVulkan_RemoveTexture(descriptorSets[i]);
    }

    descriptorSets.clear();
};

void MeshViewResources::shutdown(const VulkanContext& vkCtx)
{
    vkDestroySampler(vkCtx.device, sampler, nullptr);

    // ─[ Color Images ]───────────────────────────────────────────────────
    for (size_t i = 0; i < Config::renderer.maxFramesInFlight; i++) {
        vkDestroyFramebuffer(vkCtx.device, framebuffers[i], nullptr);
        vkDestroyImageView(vkCtx.device, colorImageViews[i], nullptr);
        vkDestroyImage(vkCtx.device, colorImages[i], nullptr);
        vkFreeMemory(vkCtx.device, colorImagesMemory[i], nullptr);
    }

    framebuffers.clear();
    colorImageViews.clear();
    colorImages.clear();
    colorImagesMemory.clear();

    // ─[ Depth Image ]────────────────────────────────────────────────────
    vkDestroyImageView(vkCtx.device, depthImageView, nullptr);
    vkDestroyImage(vkCtx.device, depthImage, nullptr);
    vkFreeMemory(vkCtx.device, depthImageMemory, nullptr);

    meshPipeline.shutdown(vkCtx);
}

void MeshViewResources::reset(const VulkanContext& vkCtx,
                              VkRenderPass         renderPass,
                              VkExtent2D           viewportExtent)
{
    vkDeviceWaitIdle(vkCtx.device);

    clearTextureDesc();
    shutdown(vkCtx);
    init(vkCtx, renderPass, viewportExtent);
    initTextureDesc();
}

VkDescriptorSet MeshViewResources::getMeshViewTextureDesc(uint32_t idx) const
{
    return descriptorSets.at(idx);
}

VkExtent2D    MeshViewResources::getImageExtent() const { return extent; }
VkFramebuffer MeshViewResources::getFrameBuffer(uint32_t idx) const { return framebuffers.at(idx); }
VkPipeline    MeshViewResources::getGraphicsPipeline() const
{
    return meshPipeline.getGraphicsPipeline();
}

VkPipelineLayout MeshViewResources::getGraphicsPipelineLayout() const
{
    return meshPipeline.getGraphicsPipelineLayout();
}

VkFormat MeshViewResources::findSupportedFormat(const VulkanContext&         vkCtx,
                                                const std::vector<VkFormat>& candidates,
                                                VkImageTiling                tiling,
                                                VkFormatFeatureFlags         features)
{
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(vkCtx.physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                 (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format!");
}

VkFormat MeshViewResources::findDepthFormat(const VulkanContext& vkCtx)
{
    return findSupportedFormat(
        vkCtx,
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}
