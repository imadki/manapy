#pragma once

#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>
#include <stdint.h>
#include <array>

namespace Config {

inline constexpr struct {
    int         width  = 1280;
    int         height = 720;
    const char* title  = "Manapy GUI";
} window;

inline constexpr struct {
    uint32_t vulkanApiVersion  = VK_API_VERSION_1_3;
    int      maxFramesInFlight = 2;

    std::array<const char*, 2> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                   VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME};

    std::array<const char*, 1> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
    bool enableValidationLayers = false;
#else
    bool enableValidationLayers = true;
#endif
} renderer;

inline constexpr struct {
    float     defaultMeshSize = 2.f;
    glm::vec3 worldMeshAnchor = {0.f, 0.f, 0.f};
    glm::vec3 worldUp         = {0.f, 0.f, 1.f};
} mesh;

// inline constexpr struct {
//     float cameraMoveSpeed = 2.5f;
//     float cameraLookSpeed = 0.1f;
// } input;
//
// inline constexpr struct {
//     bool showDebugConsole = true;
//     bool dockspaceEnabled = true;
// } ui;

} // namespace Config
