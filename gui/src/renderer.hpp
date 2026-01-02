#pragma once

#include "common.hpp"
#include "rendererUtils.hpp"
#include "window.hpp"

class Renderer {
  public:
    Renderer(const char* appName, Window& window);
    ~Renderer();

    void drawFrame();

  private:
    const std::vector<const char*> validationLayers{"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    const size_t maxFramesInFlight = 2;

  private:
#ifdef NDEBUG
    static constexpr bool enableValidationLayers = false;
#else
    static constexpr bool enableValidationLayers = true;
#endif

    Window& window;

    bool     framebufferResized = false;
    uint32_t currentFrame       = 0;

    VkInstance               vkInstance     = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;

    QueueFamilyIndices queueFamilyIndices;
    VkQueue            graphicsQueue = VK_NULL_HANDLE;
    VkQueue            presentQueue  = VK_NULL_HANDLE;

    VkSurfaceKHR   surface   = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;

    std::vector<VkImage>       swapchainImages;
    VkFormat                   swapchainImageFormat;
    VkExtent2D                 swapchainExtent;
    std::vector<VkImageView>   swapchainImageViews;
    std::vector<VkFramebuffer> swapchainFramebuffers;

    VkRenderPass     renderPass       = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       graphicsPipeline = VK_NULL_HANDLE;

    VkCommandPool                commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence>     inFlightFences;

  private:
    // ─[ MAIN FUNCTIONS ]─────────────────────────────────────────────────
    void createVulkanInstance(const char* appName);
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void allocateCommandBuffers();
    void createSyncObjects();

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIdx);

    void recreateSwapchain();
    void cleanupSwapchain();

  private:
    // ─[ HELPER FUNCTIONS ]───────────────────────────────────────────────
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

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    VkSurfaceFormatKHR
    chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR
    chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

    VkShaderModule createShaderModule(const char* bytecodePath);
};
