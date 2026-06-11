#pragma once

#include <vulkan/vulkan.h>
#include "GLFW/glfw3.h"
#include "../graphics/rendererUtils.hpp"

class VulkanDevice {
  public:
    void init(GLFWwindow* glfwWindow);
    void shutdown();

    void populate(VulkanContext* vkCtx);

  private:
    VkInstance               vkInstance     = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;

    QueueFamilyIndices queueFamilyIndices;
    VkQueue            graphicsQueue = VK_NULL_HANDLE;
    VkQueue            presentQueue  = VK_NULL_HANDLE;

    VkSurfaceKHR surface = VK_NULL_HANDLE;

  private:
    void createVulkanInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* glfwWindow);
    void pickPhysicalDevice();
    void createLogicalDevice();

  private:
    bool                     checkValidationLayersSupport();
    std::vector<const char*> getRequiredExtensions();

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT             messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                  void*                                       pUserData);

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

    static VkResult
                CreateDebugUtilsMessengerEXT(VkInstance                                instance,
                                             const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                             const VkAllocationCallbacks*              pAllocator,
                                             VkDebugUtilsMessengerEXT*                 pDebugMessenger);
    static void DestroyDebugUtilsMessengerEXT(VkInstance                   instance,
                                              VkDebugUtilsMessengerEXT     debugMessenger,
                                              const VkAllocationCallbacks* pAllocator);

    int                     rateDeviceSuitability(VkPhysicalDevice dev);
    QueueFamilyIndices      findQueueFamilies(VkPhysicalDevice dev);
    bool                    checkDeviceExtensionSupport(VkPhysicalDevice dev);
    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice dev);

    PFN_vkCmdSetPolygonModeEXT vkCmdSetPolygonModeEXT = nullptr;
};
