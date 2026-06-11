#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include "GLFW/glfw3.h"

#include "rendererUtils.hpp"

class Swapchain {
  public:
    void init(const VulkanContext& vkCtx, GLFWwindow* glfwWindow);
    void initFramebuffers(const VulkanContext& vkCtx, VkRenderPass renderPass);
    void shutdown(const VulkanContext& vkCtx);

    void populate(VulkanContext* vkCtx);

    bool acquireNextImage(const VulkanContext& vkCtx,
                          VkSemaphore          semaphore,
                          VkFence              fence,
                          uint32_t*            imageIdx);

    VkFramebuffer getFramebuffer(uint32_t idx) const;
    VkExtent2D    getExtent() const;

    bool present(const VulkanContext& vkCtx,
                 uint32_t             waitSemaphoreCount,
                 const VkSemaphore*   waitSemaphores,
                 const uint32_t*      imageIdx);

    void reset(const VulkanContext& vkCtx, GLFWwindow* glfwWindow, VkRenderPass renderPass);

  private:
    VkSwapchainKHR             swapchain = VK_NULL_HANDLE;
    std::vector<VkImage>       swapchainImages;
    std::vector<VkImageView>   swapchainImageViews;
    VkFormat                   swapchainImageFormat;
    VkExtent2D                 swapchainExtent;
    std::vector<VkFramebuffer> swapchainFramebuffers;

  private:
    void createSwapchain(const VulkanContext& vkCtx, GLFWwindow* glfwWindow);
    void createSwapchainImageViews(const VulkanContext& vkCtx);

  private:
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                GLFWwindow*                     glfwWindow);
    VkSurfaceFormatKHR
    chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR
    chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
};
