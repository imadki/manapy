#include "./uiBackend.hpp"
#include "../common/config.hpp"
#include <cstdint>

void UIBackend::init(const VulkanContext&          vkCtx,
                     GLFWwindow*                   glfwWindow,
                     ImGui_ImplVulkan_PipelineInfo imGuiPipelineInfo)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);

    ImGui_ImplVulkan_InitInfo initInfo{
        .ApiVersion         = Config::renderer.vulkanApiVersion,
        .Instance           = vkCtx.instance,
        .PhysicalDevice     = vkCtx.physicalDevice,
        .Device             = vkCtx.device,
        .QueueFamily        = vkCtx.queueFamilies.graphicsFamily.value(),
        .Queue              = vkCtx.graphicsQueue,
        .DescriptorPoolSize = 64, // TODO: calibrate pool size
        .MinImageCount      = (uint32_t)vkCtx.swapchainImageCount,
        .ImageCount         = (uint32_t)vkCtx.swapchainImageCount,
        .PipelineCache      = nullptr,
        .PipelineInfoMain   = imGuiPipelineInfo,
        .Allocator          = nullptr,
        .CheckVkResultFn =
            [](VkResult err) {
                if (err != VK_SUCCESS) {
                    throw std::runtime_error("ImGui internal Vulkan call failed with " +
                                             std::string(string_VkResult(err)));
                }
            },
    };

    ImGui_ImplVulkan_Init(&initInfo);
}

void UIBackend::shutdown()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UIBackend::newFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}
