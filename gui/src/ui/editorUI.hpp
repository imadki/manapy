#pragma once

#include "./uiBackend.hpp"
#include "./uiState.hpp"
#include <string_view>

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
    enum class Panel {
        Simulation,
        Domain,
        Viewport,
        ViewportSettings,
        Console,
    };

  private:
    UIBackend backend;
    UIState   state;

    ImGui::FileBrowser meshFileDialog;

  private:
    void buildDockLayout();
    void buildDockPanel(Panel panel);

  private:
    static void style();
    static void helpMarker(const char* desc);

    static constexpr std::string_view getLabel(Panel panel);
};
