#pragma once

#include "./uiBackend.hpp"
#include "./uiState.hpp"

class EditorUI {
  public:
    void init(const VulkanContext&          vkCtx,
              GLFWwindow*                   glfwWindow,
              ImGui_ImplVulkan_PipelineInfo imGuiPipelineInfo);
    void shutdown();

    void build();
    void insertMeshViewTexture(VkDescriptorSet meshViewTexture);

    const UIState& getState() const;

  private:
    UIBackend backend;
    UIState   state;

    ImGui::FileBrowser meshFileDialog;

  private:
    static void style();
    static void helpMarker(const char* desc);
};
