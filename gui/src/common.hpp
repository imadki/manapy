#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vk_enum_string_helper.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <gmsh.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                                                \
    do {                                                                                           \
        VkResult result = (x);                                                                     \
        if (result != VK_SUCCESS) {                                                                \
            throw std::runtime_error(#x " failed with " + std::string(string_VkResult(result)));   \
        }                                                                                          \
    } while (0)

constexpr glm::vec3 worldUp{0.f, 0.f, 1.f};
constexpr glm::vec3 modelPivot{0.f, 0.f, 0.f}; // Model world pos and camera orbit center
