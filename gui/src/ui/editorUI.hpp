#pragma once

#include "../common.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>
#include <imfilebrowser.h>

class EditorUI {
  public:
    void init(ImGui_ImplVulkan_InitInfo* initInfo, GLFWwindow* glfwWindow);
    void cleanup();

    void draw(VkCommandBuffer commandBuffer, VkDescriptorSet meshViewTexture);

    VkExtent2D getMeshViewportExtent() const;
    bool       isMeshViewportFocused() const;
    bool       isMeshViewportHovered() const;

    bool hasSelectedMesh(std::string* filePath) const;
    void clearMeshSelection();

  private:
    ImVec2 meshViewportSize{800, 600};
    bool   meshViewportFocused = false;
    bool   meshViewportHovered = false;

    ImGui::FileBrowser meshFileDialog;

  private:
    void        build(VkDescriptorSet meshViewTexture);
    static void style();
    static void helpMarker(const char* desc);
};
