#pragma once

#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "../common/config.hpp"
#include "./rendererUtils.hpp"
#include "./vulkanDevice.hpp"
#include "./swapchain.hpp"
#include "./renderPass/renderPassManager.hpp"
#include "../platform/window.hpp"
#include "../ui/uiState.hpp"
#include "../scene/cameraData.hpp"

#include <vector>

class Renderer {
  public:
    void init(GLFWwindow* glfwWindow);
    void attach(Window& window);

    void initMeshViewTextureDesc();
    void clearMeshViewTextureDesc();

    void deviceWaitIdle();
    void shutdown();

    void update(const UIState& uiState);

    const VulkanContext&          getVulkanContext() const;
    ImGui_ImplVulkan_PipelineInfo getImGuiPipelineInfo() const;

    bool            beginFrame();
    VkDescriptorSet getMeshViewTextureDesc() const;

    void drawFrame(const UIState& uiState, const CameraData& cameraData, const MeshData& meshData);

  private:
    VulkanContext vkCtx;

    GLFWwindow* glfwWindow;
    bool        isWindowResized = false;

    uint32_t currFrameIdx = 0;
    uint32_t currImageIdx;

    VulkanDevice      vulkanDevice;
    Swapchain         swapchain;
    RenderPassManager renderPassManager;

    VkCommandPool                graphicsCommandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> graphicsCommandBuffers;

    std::vector<VkSemaphore>                                  renderFinishedSemaphores;
    std::array<VkSemaphore, Config::render.maxFramesInFlight> imageAvailableSemaphores;
    std::array<VkFence, Config::render.maxFramesInFlight>     frameInFlightFences;
    std::vector<VkFence>                                      imageLastUsedFences;

  private:
    void createCommandPools();
    void initMeshManager();
    void initEditorUI();
    void allocateCommandBuffers();
    void createFrameSyncObjects();

    void recreateSwapchain();

    void resetFrameSyncObjects();

  private:
    void onWindowResize(int width, int height);

    PushConstantData getPushConstantData(const UIState&    uiState,
                                         const CameraData& cameraData,
                                         const MeshData&   meshData);
};
