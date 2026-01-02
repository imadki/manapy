#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vk_enum_string_helper.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
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
