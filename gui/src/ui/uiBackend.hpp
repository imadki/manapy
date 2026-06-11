#pragma once

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>
#include <imfilebrowser.h>
#include <vulkan/vulkan_core.h>
#include "../graphics/rendererUtils.hpp"

class UIBackend {
  public:
    void init(const VulkanContext&          vkCtx,
              GLFWwindow*                   glfwWindow,
              ImGui_ImplVulkan_PipelineInfo imGuiPipelineInfo);
    void shutdown();

    void newFrame();
};
