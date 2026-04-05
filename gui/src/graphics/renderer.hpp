#pragma once

#include "../common.hpp"
#include "../core/window.hpp"
#include "../ui/editorUI.hpp"
#include "../resources/meshManager.hpp"
#include "camera.hpp"
#include "rendererUtils.hpp"

static constexpr uint32_t VULKAN_API_VERSION = VK_API_VERSION_1_3;

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

    Window&     window;
    EditorUI    editorUI;
    Camera      camera;
    MeshManager meshManager;

    std::shared_ptr<bool> pFrameBufferResized;

    uint32_t currentFrame = 0;

    VkInstance               vkInstance     = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;

    QueueFamilyIndices queueFamilyIndices;
    VkQueue            graphicsQueue = VK_NULL_HANDLE;
    VkQueue            presentQueue  = VK_NULL_HANDLE;

    VkSurfaceKHR   surface   = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;

    VkRenderPass UIRenderPass       = VK_NULL_HANDLE;
    VkRenderPass meshViewRenderPass = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       graphicsPipeline = VK_NULL_HANDLE;

    std::vector<VkImage>       swapchainImages;
    std::vector<VkImageView>   swapchainImageViews;
    VkFormat                   swapchainImageFormat;
    VkExtent2D                 swapchainExtent;
    std::vector<VkFramebuffer> swapchainFramebuffers;

    struct {
        VkExtent2D extent{800, 600};
        VkFormat   format{VK_FORMAT_B8G8R8A8_UNORM};

        std::vector<VkImage>        colorImages;
        std::vector<VkDeviceMemory> colorImagesMemory;
        std::vector<VkImageView>    colorImageViews;

        VkImage        depthImage       = VK_NULL_HANDLE;
        VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
        VkImageView    depthImageView   = VK_NULL_HANDLE;

        std::vector<VkFramebuffer> framebuffers;

        std::vector<VkDescriptorSet> descriptorSets;
    } meshViewFrameData;

    VkSampler sampler = VK_NULL_HANDLE;

    VkCommandPool                graphicsCommandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> graphicsCommandBuffers;

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
    void createSwapchainImageViews();
    void createRenderPasses();
    void createGraphicsPipeline();
    void createCommandPools();
    void initMeshManager();
    void initEditorUI();
    void createSampler();
    void createMeshViewResources();
    void createSwapchainFramebuffers();
    void allocateCommandBuffers();
    void createSyncObjects();

    void recordDrawCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIdx);

    void recreateMeshViewResources();
    void cleanupMeshViewResources();

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

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                 VkImageTiling                tiling,
                                 VkFormatFeatureFlags         features);

    VkFormat findDepthFormat();

    void createImage(uint32_t              width,
                     uint32_t              height,
                     VkFormat              format,
                     VkImageTiling         tiling,
                     VkImageUsageFlags     usage,
                     VkMemoryPropertyFlags properties,
                     VkImage&              image,
                     VkDeviceMemory&       imageMemory);
    void createImageView(VkImage            image,
                         VkFormat           format,
                         VkImageAspectFlags aspectFlags,
                         VkImageView&       imageView);

    void             updateCamera();
    PushConstantData getPushConstantData();
};
