#include "rendererUtils.hpp"

// ╭─────────────────────────────────────────────────────────╮
// │                    Helper Structures                    │
// ╰─────────────────────────────────────────────────────────╯

bool QueueFamilyIndices::isComplete()
{
    return graphicsFamily.has_value() && presentFamily.has_value();
}

bool SwapchainSupportDetails::isAdequate() { return !formats.empty() && !presentModes.empty(); }

VkVertexInputBindingDescription Vertex::getBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions;

    // ─[ Position ]───────────────────────────────────────────────────────
    attributeDescriptions[0] = {
        .location = 0,
        .binding  = 0,
        .format   = VK_FORMAT_R32G32B32_SFLOAT,
        .offset   = offsetof(Vertex, position),
    };

    // ─[ Color ]──────────────────────────────────────────────────────────
    attributeDescriptions[1] = {
        .location = 1,
        .binding  = 0,
        .format   = VK_FORMAT_R32G32B32_SFLOAT,
        .offset   = offsetof(Vertex, color),
    };

    return attributeDescriptions;
}

// ╭─────────────────────────────────────────────────────────╮
// │                     Helper Funtions                     │
// ╰─────────────────────────────────────────────────────────╯

void utils::createImage(VkPhysicalDevice      physicalDevice,
                        VkDevice              device,
                        uint32_t              width,
                        uint32_t              height,
                        VkFormat              format,
                        VkImageTiling         tiling,
                        VkImageUsageFlags     usage,
                        VkMemoryPropertyFlags properties,
                        VkImage*              image,
                        VkDeviceMemory*       imageMemory)
{
    // ─[ Create Image Object ]────────────────────────────────────────────
    VkImageCreateInfo imageInfo{
        .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format    = format,
        .extent{
            .width  = width,
            .height = height,
            .depth  = 1,
        },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = tiling,
        .usage         = usage,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, image));

    // ─[ Allocate Image Memory ]──────────────────────────────────────────
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, *image, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{
        .sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex =
            utils::findMemoryType(memoryRequirements.memoryTypeBits, properties, physicalDevice),
    };

    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, imageMemory));

    VK_CHECK(vkBindImageMemory(device, *image, *imageMemory, 0));
}

void utils::createImageView(VkDevice           device,
                            VkImage            image,
                            VkFormat           format,
                            VkImageAspectFlags aspectFlags,
                            VkImageView*       imageView)
{
    VkImageViewCreateInfo viewInfo{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = format,
        .subresourceRange{
            .aspectMask     = aspectFlags,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };

    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, imageView));
}

void utils::createBuffer(VkDeviceSize          size,
                         VkBufferUsageFlags    usage,
                         VkMemoryPropertyFlags properties,
                         VkBuffer*             buffer,
                         VkDeviceMemory*       bufferMemory,
                         VkPhysicalDevice      physicalDevice,
                         VkDevice              device)
{
    // ─[ Create Buffer Object ]───────────────────────────────────────────
    VkBufferCreateInfo bufferInfo{
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, buffer));

    // ─[ Allocate Buffer Memory ]─────────────────────────────────────────
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{
        .sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex =
            findMemoryType(memoryRequirements.memoryTypeBits, properties, physicalDevice),
    };

    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, bufferMemory));

    VK_CHECK(vkBindBufferMemory(device, *buffer, *bufferMemory, 0));
}

void utils::copyBuffer(VkBuffer      srcBuffer,
                       VkBuffer      dstBuffer,
                       VkDeviceSize  size,
                       VkDevice      device,
                       VkCommandPool commandPool,
                       VkQueue       queue)
{
    // ─[ Allocate Transfer Command Buffer ]───────────────────────────────
    VkCommandBufferAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = commandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));

    // ─[ Record Commands ]────────────────────────────────────────────────
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkBufferCopy copyRegion{
        .size = size,
    };

    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    // ─[ Submit Command Buffer ]──────────────────────────────────────────
    VkSubmitInfo submitInfo{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &commandBuffer,
    };

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    // ─[ Cleanup ]────────────────────────────────────────────────────────
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

uint32_t utils::findMemoryType(uint32_t              typeFilter,
                               VkMemoryPropertyFlags properties,
                               VkPhysicalDevice      physicalDevice)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}
