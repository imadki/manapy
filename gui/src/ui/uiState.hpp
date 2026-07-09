#pragma once

#include <filesystem>
#include <imgui.h>
#include <vulkan/vulkan_core.h>

struct UIState {
    struct {
        ImVec2 size = {0, 0};

        bool isFocused = false;
        bool isHovered = false;

        float cameraFov = 60.f;

        ImVec2 orbitSpeed = {.8f, .6f};
        float  zoomSpeed  = .5f;

        VkPolygonMode polygoneMode = VK_POLYGON_MODE_FILL;
    } meshView;

    struct {
        bool                  isSelected = false;
        std::filesystem::path path;
    } meshFile;
};
