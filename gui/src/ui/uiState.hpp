#pragma once

#include <imgui.h>
#include <string>
#include <vulkan/vulkan_core.h>

struct UIState {
    struct {
        ImVec2 size = {0, 0};

        bool isFocused = false;
        bool isHovered = false;

        VkPolygonMode polygoneMode = VK_POLYGON_MODE_FILL;
    } meshView;

    struct {
        bool        isSelected = false;
        std::string path;
    } meshFile;
};
