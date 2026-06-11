#include "swapchain.hpp"
#include "GLFW/glfw3.h"
#include "rendererUtils.hpp"
#include <algorithm>
#include <vulkan/vulkan_core.h>

void Swapchain::init(const VulkanContext& vkCtx, GLFWwindow* glfwWindow)
{
    createSwapchain(vkCtx, glfwWindow);
    createSwapchainImageViews(vkCtx);
}

void Swapchain::initFramebuffers(const VulkanContext& vkCtx, VkRenderPass renderPass)
{
    swapchainFramebuffers.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        std::array<VkImageView, 1> attachments = {
            swapchainImageViews[i],
        };

        VkFramebufferCreateInfo createInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = renderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .width           = swapchainExtent.width,
            .height          = swapchainExtent.height,
            .layers          = 1,
        };

        VK_CHECK(
            vkCreateFramebuffer(vkCtx.device, &createInfo, nullptr, &swapchainFramebuffers[i]));
    }
}

void Swapchain::shutdown(const VulkanContext& vkCtx)
{
    for (size_t i = 0; i < swapchainImages.size(); ++i) {
        vkDestroyImageView(vkCtx.device, swapchainImageViews[i], nullptr);

        vkDestroyFramebuffer(vkCtx.device, swapchainFramebuffers[i], nullptr);
    }

    vkDestroySwapchainKHR(vkCtx.device, swapchain, nullptr);
}

void Swapchain::populate(VulkanContext* vkCtx)
{
    vkCtx->swapchainImageCount  = swapchainImages.size();
    vkCtx->swapchainImageFormat = swapchainImageFormat;
}

bool Swapchain::acquireNextImage(const VulkanContext& vkCtx,
                                 VkSemaphore          semaphore,
                                 VkFence              fence,
                                 uint32_t*            imageIdx)
{
    VkResult res =
        vkAcquireNextImageKHR(vkCtx.device, swapchain, UINT64_MAX, semaphore, fence, imageIdx);

    // Check for swapchain expiry
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        return false;
    }
    else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image!");
    }

    return true;
}

VkFramebuffer Swapchain::getFramebuffer(uint32_t idx) const
{
    return swapchainFramebuffers.at(idx);
}

VkExtent2D Swapchain::getExtent() const { return swapchainExtent; }

bool Swapchain::present(const VulkanContext& vkCtx,
                        uint32_t             waitSemaphoreCount,
                        const VkSemaphore*   waitSemaphores,
                        const uint32_t*      imageIdx)
{

    VkSwapchainKHR swapchains[] = {swapchain};

    VkPresentInfoKHR presentInfo{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = waitSemaphoreCount,
        .pWaitSemaphores    = waitSemaphores,
        .swapchainCount     = 1,
        .pSwapchains        = swapchains,
        .pImageIndices      = imageIdx,
    };

    VkResult res = vkQueuePresentKHR(vkCtx.presentQueue, &presentInfo);

    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        return false;
    }
    else if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image!");
    }

    return true;
}

void Swapchain::reset(const VulkanContext& vkCtx, GLFWwindow* glfwWindow, VkRenderPass renderPass)
{
    // Handle minimization
    int width = 0, height = 0;
    glfwGetFramebufferSize(glfwWindow, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(glfwWindow, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(vkCtx.device);

    shutdown(vkCtx);

    createSwapchain(vkCtx, glfwWindow);
    createSwapchainImageViews(vkCtx);

    initFramebuffers(vkCtx, renderPass);
}

void Swapchain::createSwapchain(const VulkanContext& vkCtx, GLFWwindow* glfwWindow)
{
    SwapchainSupportDetails support = vkCtx.swapchainSupport;

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    VkPresentModeKHR   presentMode   = chooseSwapPresentMode(support.presentModes);
    VkExtent2D         extent        = chooseSwapExtent(support.capabilities, glfwWindow);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    // 'maxImageCount == 0' is a special value to indicate that there is no maximum
    if (support.capabilities.maxImageCount && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = vkCtx.surface,
        .minImageCount    = imageCount,
        .imageFormat      = surfaceFormat.format,
        .imageColorSpace  = surfaceFormat.colorSpace,
        .imageExtent      = extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = support.capabilities.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = presentMode,
        .clipped          = VK_TRUE,
        .oldSwapchain     = VK_NULL_HANDLE,
    };

    uint32_t indicesArray[] = {
        vkCtx.queueFamilies.graphicsFamily.value(),
        vkCtx.queueFamilies.presentFamily.value(),
    };

    // Check for unique queue families
    if (vkCtx.queueFamilies.graphicsFamily != vkCtx.queueFamilies.presentFamily) {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = indicesArray;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VK_CHECK(vkCreateSwapchainKHR(vkCtx.device, &createInfo, nullptr, &swapchain));

    // ─[ Save Swapchain Data ]───────────────────────────────────────────
    vkGetSwapchainImagesKHR(vkCtx.device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(vkCtx.device, swapchain, &imageCount, swapchainImages.data());

    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent      = extent;
}

void Swapchain::createSwapchainImageViews(const VulkanContext& vkCtx)
{
    swapchainImageViews.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        utils::createImageView(vkCtx.device,
                               swapchainImages[i],
                               swapchainImageFormat,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               &swapchainImageViews[i]);
    }
}

VkSurfaceFormatKHR
Swapchain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR
Swapchain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    return VK_PRESENT_MODE_FIFO_KHR;

    // for (const auto& availablePresentMode : availablePresentModes) {
    //     if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
    //         return availablePresentMode;
    //     }
    // }
    //
    // return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Swapchain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                       GLFWwindow*                     glfwWindow)
{
    // Special value to indicate that the extent should be chosen and set manually
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    int width, height;
    glfwGetFramebufferSize(glfwWindow, &width, &height);

    VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    actualExtent.width = std::clamp(actualExtent.width,
                                    capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);

    actualExtent.height = std::clamp(actualExtent.height,
                                     capabilities.minImageExtent.height,
                                     capabilities.maxImageExtent.height);

    return actualExtent;
}
