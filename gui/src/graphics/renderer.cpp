#include "renderer.hpp"
#include "rendererUtils.hpp"

#include <cstdint>
#include <stdbool.h>

// TODO: Consider multisampling

void Renderer::init(GLFWwindow* glfwWindow)
{
    this->glfwWindow = glfwWindow;

    vulkanDevice.init(glfwWindow);
    vulkanDevice.populate(&vkCtx);

    swapchain.init(vkCtx, glfwWindow);
    swapchain.populate(&vkCtx);

    renderPassManager.init(vkCtx, glfwWindow);

    createCommandPools();
    vkCtx.graphicsCommandPool = graphicsCommandPool;

    swapchain.initFramebuffers(vkCtx, renderPassManager.getRenderPass(UI));

    allocateCommandBuffers();
    createFrameSyncObjects();
}

void Renderer::attach(Window& window)
{
    window.setResizeExtCallback([this](int width, int height) { onWindowResize(width, height); });
}

void Renderer::initMeshViewTextureDesc() { renderPassManager.initMeshViewTextureDesc(); }
void Renderer::clearMeshViewTextureDesc() { renderPassManager.clearMeshViewTextureDesc(); }

void Renderer::deviceWaitIdle() { vkDeviceWaitIdle(vkCtx.device); }

void Renderer::shutdown()
{
    vkDeviceWaitIdle(vkCtx.device);

    renderPassManager.shutdown(vkCtx);
    swapchain.shutdown(vkCtx);

    for (size_t i = 0; i < Config::render.maxFramesInFlight; i++) {
        vkDestroySemaphore(vkCtx.device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(vkCtx.device, frameInFlightFences[i], nullptr);
    }

    for (size_t i = 0; i < vkCtx.swapchainImageCount; ++i) {
        vkDestroySemaphore(vkCtx.device, renderFinishedSemaphores[i], nullptr);
    }

    vkDestroyCommandPool(vkCtx.device, graphicsCommandPool, nullptr);

    vulkanDevice.shutdown();
}

const VulkanContext& Renderer::getVulkanContext() const { return vkCtx; }

ImGui_ImplVulkan_PipelineInfo Renderer::getImGuiPipelineInfo() const
{
    return renderPassManager.getImGuiPipelineInfo();
}

void Renderer::update(const UIState& uiState) { renderPassManager.update(vkCtx, uiState); }

void Renderer::onWindowResize(int width, int height) { isWindowResized = true; }

bool Renderer::beginFrame()
{
    VK_CHECK(
        vkWaitForFences(vkCtx.device, 1, &frameInFlightFences[currFrameIdx], VK_TRUE, UINT64_MAX));

    bool success = swapchain.acquireNextImage(vkCtx,
                                              imageAvailableSemaphores[currFrameIdx],
                                              VK_NULL_HANDLE,
                                              &currImageIdx);

    if (!success) {
        recreateSwapchain();
        return false;
    }

    if (imageLastUsedFences[currImageIdx] != VK_NULL_HANDLE) {
        VK_CHECK(vkWaitForFences(vkCtx.device,
                                 1,
                                 &imageLastUsedFences[currImageIdx],
                                 VK_TRUE,
                                 UINT64_MAX));
    }

    imageLastUsedFences[currImageIdx] = frameInFlightFences[currFrameIdx];

    VK_CHECK(vkResetFences(vkCtx.device, 1, &frameInFlightFences[currFrameIdx]));
    VK_CHECK(vkResetCommandBuffer(graphicsCommandBuffers[currFrameIdx], 0));

    return true;
}

VkDescriptorSet Renderer::getMeshViewTextureDesc() const
{
    return renderPassManager.getMeshViewTextureDesc(currFrameIdx);
}

void Renderer::drawFrame(const UIState&    uiState,
                         const CameraData& cameraData,
                         const MeshData&   meshData)
{
    // ─[ Record Draw Commands ]───────────────────────────────────────────
    PushConstantData pushConstant = getPushConstantData(uiState, cameraData, meshData);

    renderPassManager.recordDrawCommandBuffer(vkCtx,
                                              graphicsCommandBuffers[currFrameIdx],
                                              swapchain.getFramebuffer(currImageIdx),
                                              swapchain.getExtent(),
                                              currFrameIdx,
                                              currImageIdx,
                                              pushConstant,
                                              uiState,
                                              meshData);

    // ─[ Submit Command Buffer ]──────────────────────────────────────────
    VkSemaphore          waitSemaphores[]   = {imageAvailableSemaphores[currFrameIdx]};
    VkPipelineStageFlags waitStages[]       = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore          signalSemaphores[] = {renderFinishedSemaphores[currImageIdx]};

    VkSubmitInfo submitInfo{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount   = 1,
        .pWaitSemaphores      = waitSemaphores,
        .pWaitDstStageMask    = waitStages,
        .commandBufferCount   = 1,
        .pCommandBuffers      = &graphicsCommandBuffers[currFrameIdx],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores    = signalSemaphores,
    };

    VK_CHECK(vkQueueSubmit(vkCtx.graphicsQueue, 1, &submitInfo, frameInFlightFences[currFrameIdx]));

    // ─[ Present Drawn Image ]────────────────────────────────────────────
    bool succcess = swapchain.present(vkCtx, 1, signalSemaphores, &currImageIdx);

    if (!succcess || isWindowResized) {
        recreateSwapchain();
        isWindowResized = false;
    }

    currFrameIdx = (currFrameIdx + 1) % Config::render.maxFramesInFlight;
}

void Renderer::createCommandPools()
{
    // ─[ Graphics ]───────────────────────────────────────────────────────
    VkCommandPoolCreateInfo createInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = vkCtx.queueFamilies.graphicsFamily.value(),
    };

    VK_CHECK(vkCreateCommandPool(vkCtx.device, &createInfo, nullptr, &graphicsCommandPool));
}

void Renderer::allocateCommandBuffers()
{
    // ─[ Graphics ]───────────────────────────────────────────────────────
    graphicsCommandBuffers.resize(Config::render.maxFramesInFlight);

    VkCommandBufferAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = graphicsCommandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<uint32_t>(graphicsCommandBuffers.size()),
    };

    VK_CHECK(vkAllocateCommandBuffers(vkCtx.device, &allocInfo, graphicsCommandBuffers.data()));
}

void Renderer::createFrameSyncObjects()
{
    VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    VkFenceCreateInfo fenceInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (size_t i = 0; i < Config::render.maxFramesInFlight; i++) {
        VK_CHECK(
            vkCreateSemaphore(vkCtx.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]));
        VK_CHECK(vkCreateFence(vkCtx.device, &fenceInfo, nullptr, &frameInFlightFences[i]));
    }

    imageLastUsedFences.resize(vkCtx.swapchainImageCount);
    imageLastUsedFences.assign(vkCtx.swapchainImageCount, VK_NULL_HANDLE);

    renderFinishedSemaphores.resize(vkCtx.swapchainImageCount);

    for (size_t i = 0; i < vkCtx.swapchainImageCount; ++i)
        VK_CHECK(
            vkCreateSemaphore(vkCtx.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]));
}

PushConstantData Renderer::getPushConstantData(const UIState&    uiState,
                                               const CameraData& cameraData,
                                               const MeshData&   meshData)
{
    glm::mat4 projMat = cameraData.projectionMatrix;
    projMat[1][1] *= -1.f;

    return PushConstantData{.mvpMatrix = projMat * cameraData.viewMatrix * meshData.modelMatrix};
}

void Renderer::recreateSwapchain()
{
    swapchain.reset(vkCtx, glfwWindow, renderPassManager.getRenderPass(UI));
    swapchain.populate(&vkCtx);

    resetFrameSyncObjects();

    ImGui_ImplVulkan_SetMinImageCount(vkCtx.swapchainImageCount);
}

void Renderer::resetFrameSyncObjects()
{
    for (size_t i = 0; i < vkCtx.swapchainImageCount; ++i) {
        vkDestroySemaphore(vkCtx.device, renderFinishedSemaphores[i], nullptr);
    }

    imageLastUsedFences.resize(vkCtx.swapchainImageCount);
    imageLastUsedFences.assign(vkCtx.swapchainImageCount, VK_NULL_HANDLE);

    renderFinishedSemaphores.resize(vkCtx.swapchainImageCount);

    VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    for (size_t i = 0; i < vkCtx.swapchainImageCount; ++i)
        VK_CHECK(
            vkCreateSemaphore(vkCtx.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]));
}
