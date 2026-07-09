#include "vulkanDevice.hpp"
#include "../common/config.hpp"
#include "rendererUtils.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <set>
#include <map>

// TODO: Add an initialization guard (e.g. a bool flag) to ensure `init()`
// has been called before any dependent function. Throw an error otherwise.

void VulkanDevice::init(GLFWwindow* glfwWindow)
{
    createVulkanInstance();
    setupDebugMessenger();
    createSurface(glfwWindow);
    pickPhysicalDevice();
    createLogicalDevice();
}

void VulkanDevice::populate(VulkanContext* vkCtx)
{
    vkCtx->instance               = vkInstance;
    vkCtx->physicalDevice         = physicalDevice;
    vkCtx->device                 = device;
    vkCtx->surface                = surface;
    vkCtx->vkCmdSetPolygonModeEXT = vkCmdSetPolygonModeEXT;
    vkCtx->graphicsQueue          = graphicsQueue;
    vkCtx->presentQueue           = presentQueue;
    vkCtx->queueFamilies          = queueFamilyIndices;
    vkCtx->swapchainSupport       = querySwapchainSupport(physicalDevice);
}

void VulkanDevice::shutdown()
{
    vkDestroyDevice(device, nullptr);

    if constexpr (Config::render.enableValidationLayers)
        DestroyDebugUtilsMessengerEXT(vkInstance, debugMessenger, nullptr);

    vkDestroySurfaceKHR(vkInstance, surface, nullptr);
    vkDestroyInstance(vkInstance, nullptr);
}

void VulkanDevice::createVulkanInstance()
{
    VkApplicationInfo appInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = Config::window.title,
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName        = "N/A",
        .engineVersion      = VK_MAKE_VERSION(0, 0, 0),
        .apiVersion         = Config::render.vulkanApiVersion,
    };

    VkInstanceCreateInfo createInfo{
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };

    // ─[ Extensions ]─────────────────────────────────────────────────────
    std::vector<const char*> extensions = getRequiredExtensions();

    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // ─[ Validation Layers ]──────────────────────────────────────────────
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if constexpr (Config::render.enableValidationLayers) {
        if (!checkValidationLayersSupport()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        createInfo.enabledLayerCount =
            static_cast<uint32_t>(Config::render.validationLayers.size());
        createInfo.ppEnabledLayerNames = Config::render.validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else {
        createInfo.enabledLayerCount   = 0;
        createInfo.ppEnabledLayerNames = nullptr;
    }

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &vkInstance));
}

void VulkanDevice::setupDebugMessenger()
{
    if constexpr (!Config::render.enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    VK_CHECK(CreateDebugUtilsMessengerEXT(vkInstance, &createInfo, nullptr, &debugMessenger));
}

void VulkanDevice::createSurface(GLFWwindow* glfwWindow)
{
    VK_CHECK(glfwCreateWindowSurface(vkInstance, glfwWindow, nullptr, &surface));
}

void VulkanDevice::pickPhysicalDevice()
{
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(vkInstance, &deviceCount, nullptr);

    if (!deviceCount) throw std::runtime_error("Failed to find GPU with Vulkan support!");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(vkInstance, &deviceCount, devices.data());

    // Rank available devices
    std::multimap<int, VkPhysicalDevice> candidates;

    for (const auto& device : devices) {
        int score = rateDeviceSuitability(device);
        candidates.insert(std::make_pair(score, device));
    }

    if (candidates.rbegin()->first >= 0)
        physicalDevice = candidates.rbegin()->second;
    else
        throw std::runtime_error("Failed to find suitable GPU!");

    queueFamilyIndices = findQueueFamilies(physicalDevice);
}

void VulkanDevice::createLogicalDevice()
{
    std::set<uint32_t> uniqueQueueFamilies = {queueFamilyIndices.graphicsFamily.value(),
                                              queueFamilyIndices.presentFamily.value()};

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};

    float queuePriority = 1.0f;
    for (uint32_t queueFamilyIndex : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndex,
            .queueCount       = 1,
            .pQueuePriorities = &queuePriority,
        };

        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures enabledFeatures{
        .fillModeNonSolid = VK_TRUE,
    };

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
        .extendedDynamicState3PolygonMode = VK_TRUE,
    };

    VkDeviceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &eds3Features,
        .queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos       = queueCreateInfos.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(Config::render.deviceExtensions.size()),
        .ppEnabledExtensionNames = Config::render.deviceExtensions.data(),
        .pEnabledFeatures        = &enabledFeatures,
    };

    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));

    vkCmdSetPolygonModeEXT = reinterpret_cast<PFN_vkCmdSetPolygonModeEXT>(
        vkGetDeviceProcAddr(device, "vkCmdSetPolygonModeEXT"));

    if (!vkCmdSetPolygonModeEXT) throw std::runtime_error("vkCmdSetPolygonModeEXT not available");

    vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
}

bool VulkanDevice::checkValidationLayersSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : Config::render.validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerProperties.layerName, layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) return false;
    }

    return true;
}

std::vector<const char*> VulkanDevice::getRequiredExtensions()
{
    uint32_t     glfwExtensionCount = 0;
    const char** glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if constexpr (Config::render.enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
VulkanDevice::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT             messageType,
                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                            void*                                       pUserData)
{
    switch (messageSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        std::cerr << "[VALIDATION] [WARNING] " << pCallbackData->pMessage << std::endl;
        break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        std::cerr << "[VALIDATION] [ERROR] " << pCallbackData->pMessage << std::endl;
        break;

    default:;
    }

    return VK_FALSE;
}

void VulkanDevice::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {
        .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };
}

VkResult
VulkanDevice::CreateDebugUtilsMessengerEXT(VkInstance                                instance,
                                           const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                           const VkAllocationCallbacks*              pAllocator,
                                           VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,
                                                                  "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr)
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void VulkanDevice::DestroyDebugUtilsMessengerEXT(VkInstance                   instance,
                                                 VkDebugUtilsMessengerEXT     debugMessenger,
                                                 const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance,
        "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) func(instance, debugMessenger, pAllocator);
}

int VulkanDevice::rateDeviceSuitability(VkPhysicalDevice dev)
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(dev, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(dev, &deviceFeatures);

    // ─[ Check Device Suitability ]───────────────────────────────────────
    bool isSuitable = findQueueFamilies(dev).isComplete() && checkDeviceExtensionSupport(dev) &&
                      querySwapchainSupport(dev).isAdequate() && deviceFeatures.fillModeNonSolid;

    if (!isSuitable) return -1;

    // ─[ Rate Device ]───────────────────────────────────────────────────
    int score = 0;

    // Discrete GPU
    score += (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1000 : 0;

    return score;
}

QueueFamilyIndices VulkanDevice::findQueueFamilies(VkPhysicalDevice dev)
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        // Check for presentation support
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);

        if (presentSupport) indices.presentFamily = i;

        if (indices.isComplete()) break;

        i++;
    }

    return indices;
}

bool VulkanDevice::checkDeviceExtensionSupport(VkPhysicalDevice dev)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(Config::render.deviceExtensions.begin(),
                                             Config::render.deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SwapchainSupportDetails VulkanDevice::querySwapchainSupport(VkPhysicalDevice dev)
{
    SwapchainSupportDetails details;

    // ─[ Capabilities ]───────────────────────────────────────────────────
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &details.capabilities);

    // ─[ Formats ]────────────────────────────────────────────────────────
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, nullptr);
    if (formatCount) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, details.formats.data());
    }

    // ─[ Present Modes ]──────────────────────────────────────────────────
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &presentModeCount, nullptr);

    if (presentModeCount) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev,
                                                  surface,
                                                  &presentModeCount,
                                                  details.presentModes.data());
    }

    return details;
}
