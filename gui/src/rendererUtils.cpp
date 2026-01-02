#include "rendererUtils.hpp"

bool QueueFamilyIndices::isComplete()
{
    return graphicsFamily.has_value() && presentFamily.has_value();
}

bool SwapchainSupportDetails::isAdequate() { return !formats.empty() && !presentModes.empty(); }
