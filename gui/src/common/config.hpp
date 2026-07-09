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
    uint32_t vulkanApiVersion = VK_API_VERSION_1_3;

    std::array<const char*, 1> validationLayers = {"VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
    bool enableValidationLayers = false;
#else
    bool enableValidationLayers = true;
#endif

    int maxFramesInFlight = 2;

    std::array<const char*, 2> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                   VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME};

    VkClearColorValue clearColor = {{.007f, .007f, .007f, 1.0f}};
} render;

inline constexpr struct {
    float     defaultMeshSize = 2.f;
    glm::vec3 worldUp         = {0.f, 0.f, 1.f};
} mesh;

} // namespace Config
