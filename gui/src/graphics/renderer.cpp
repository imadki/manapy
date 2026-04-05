#include "renderer.hpp"

Renderer::Renderer(const char* appName, Window& window) : window(window)
{
    pFrameBufferResized = window.pFrameBufferResized;

    createVulkanInstance(appName);
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createSwapchainImageViews();
    createRenderPasses();
    createGraphicsPipeline();
    createCommandPools();
    createSampler();
    initMeshManager();
    initEditorUI();
    createMeshViewResources();
    createSwapchainFramebuffers();
    allocateCommandBuffers();
    createSyncObjects();
}

Renderer::~Renderer()
{
    vkDeviceWaitIdle(device);

    meshManager.cleanup();
    cleanupMeshViewResources();
    editorUI.cleanup();
    cleanupSwapchain();

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, graphicsCommandPool, nullptr);

    vkDestroySampler(device, sampler, nullptr);

    vkDestroyRenderPass(device, meshViewRenderPass, nullptr);
    vkDestroyRenderPass(device, UIRenderPass, nullptr);

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    vkDestroyDevice(device, nullptr);

    if constexpr (enableValidationLayers)
        DestroyDebugUtilsMessengerEXT(vkInstance, debugMessenger, nullptr);

    vkDestroySurfaceKHR(vkInstance, surface, nullptr);
    vkDestroyInstance(vkInstance, nullptr);
}

// ╭─────────────────────────────────────────────────────────╮
// │                     MAIN FUNCTIONS                      │
// ╰─────────────────────────────────────────────────────────╯
void Renderer::drawFrame()
{
    VK_CHECK(vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX));

    uint32_t imageIdx;
    VkResult res = vkAcquireNextImageKHR(device,
                                         swapchain,
                                         UINT64_MAX,
                                         imageAvailableSemaphores[currentFrame],
                                         VK_NULL_HANDLE,
                                         &imageIdx);

    // Check for swapchain expiry
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image!");
    }

    VK_CHECK(vkResetFences(device, 1, &inFlightFences[currentFrame]));

    VK_CHECK(vkResetCommandBuffer(graphicsCommandBuffers[currentFrame], 0));

    // Check for mesh viewport extent change
    VkExtent2D viewportExtent = editorUI.getMeshViewportExtent();
    if (meshViewFrameData.extent.width != viewportExtent.width ||
        meshViewFrameData.extent.height != viewportExtent.height) {
        meshViewFrameData.extent = viewportExtent;
        recreateMeshViewResources();
    }

    // Check for new mesh selection
    std::string meshFilePath;
    if (editorUI.hasSelectedMesh(&meshFilePath)) {
        meshManager.loadMesh(meshFilePath);
        editorUI.clearMeshSelection();
    }

    // ─[ Record Draw Commands ]───────────────────────────────────────────
    recordDrawCommandBuffer(graphicsCommandBuffers[currentFrame], imageIdx);

    // ─[ Submit Command Buffer ]──────────────────────────────────────────
    VkSemaphore          waitSemaphores[]   = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[]       = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore          signalSemaphores[] = {renderFinishedSemaphores[imageIdx]};

    VkSubmitInfo submitInfo{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount   = 1,
        .pWaitSemaphores      = waitSemaphores,
        .pWaitDstStageMask    = waitStages,
        .commandBufferCount   = 1,
        .pCommandBuffers      = &graphicsCommandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores    = signalSemaphores,
    };

    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));

    // ─[ Present Drawn Image ]────────────────────────────────────────────
    VkSwapchainKHR swapchains[] = {swapchain};

    VkPresentInfoKHR presentInfo{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = signalSemaphores,
        .swapchainCount     = 1,
        .pSwapchains        = swapchains,
        .pImageIndices      = &imageIdx,
    };

    res = vkQueuePresentKHR(presentQueue, &presentInfo);

    // Check for framebuffer resize
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || *pFrameBufferResized) {
        *pFrameBufferResized = false;
        recreateSwapchain();
    }
    else if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image!");
    }

    currentFrame = (currentFrame + 1) % maxFramesInFlight;
}

void Renderer::createVulkanInstance(const char* appName)
{
    VkApplicationInfo appInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = appName,
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName        = "N/A",
        .engineVersion      = VK_MAKE_VERSION(0, 0, 0),
        .apiVersion         = VULKAN_API_VERSION,
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
    if constexpr (enableValidationLayers) {
        if (!checkValidationLayersSupport()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else {
        createInfo.enabledLayerCount   = 0;
        createInfo.ppEnabledLayerNames = nullptr;
    }

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &vkInstance));
}

void Renderer::setupDebugMessenger()
{
    if constexpr (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    VK_CHECK(CreateDebugUtilsMessengerEXT(vkInstance, &createInfo, nullptr, &debugMessenger));
}

void Renderer::createSurface() { window.createSurface(vkInstance, &surface); }

void Renderer::pickPhysicalDevice()
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

void Renderer::createLogicalDevice()
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

    VkDeviceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos       = queueCreateInfos.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures        = nullptr,
    };

    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));

    vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
}

void Renderer::createSwapchain()
{
    SwapchainSupportDetails swapchainSupport = querySwapchainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
    VkPresentModeKHR   presentMode   = chooseSwapPresentMode(swapchainSupport.presentModes);
    VkExtent2D         extent        = chooseSwapExtent(swapchainSupport.capabilities);

    uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
    // 'maxImageCount == 0' is a special value to indicate that there is no maximum
    if (swapchainSupport.capabilities.maxImageCount &&
        imageCount > swapchainSupport.capabilities.maxImageCount) {
        imageCount = swapchainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = surface,
        .minImageCount    = imageCount,
        .imageFormat      = surfaceFormat.format,
        .imageColorSpace  = surfaceFormat.colorSpace,
        .imageExtent      = extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = swapchainSupport.capabilities.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = presentMode,
        .clipped          = VK_TRUE,
        .oldSwapchain     = VK_NULL_HANDLE,
    };

    uint32_t indicesArray[] = {
        queueFamilyIndices.graphicsFamily.value(),
        queueFamilyIndices.presentFamily.value(),
    };

    // Check for unique queue families
    if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily) {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = indicesArray;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain));

    // ─[ Save Swapchain Data ]───────────────────────────────────────────
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent      = extent;

    // ─[ Recreate Semaphores ]────────────────────────────────────────────
    renderFinishedSemaphores.resize(imageCount);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (size_t i = 0; i < imageCount; ++i)
        VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]));
}

void Renderer::createSwapchainImageViews()
{
    swapchainImageViews.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        createImageView(swapchainImages[i],
                        swapchainImageFormat,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        swapchainImageViews[i]);
    }
}

void Renderer::createRenderPasses()
{
    // ╭─────────────────────────────────────────────────────────╮
    // │                  Mesh View Render Pass                  │
    // ╰─────────────────────────────────────────────────────────╯
    {
        // ─[ Color Attachment ]───────────────────────────────────────────────
        VkAttachmentDescription colorAttachment{
            .format         = meshViewFrameData.format,
            .samples        = VK_SAMPLE_COUNT_1_BIT,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkAttachmentReference colorAttachmentRef{
            .attachment = 0,
            .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        // ─[ Depth Attachment ]───────────────────────────────────────────────
        VkAttachmentDescription depthAttachment{
            .format         = findDepthFormat(),
            .samples        = VK_SAMPLE_COUNT_1_BIT,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        VkAttachmentReference depthAttachmentRef{
            .attachment = 1,
            .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        // ─[ Subpass ]────────────────────────────────────────────────────────
        VkSubpassDescription subpass{
            .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount    = 1,
            .pColorAttachments       = &colorAttachmentRef,
            .pDepthStencilAttachment = &depthAttachmentRef,
        };

        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0] = {
            .srcSubpass    = VK_SUBPASS_EXTERNAL,
            .dstSubpass    = 0,
            .srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        };

        dependencies[1] = {
            .srcSubpass      = 0,
            .dstSubpass      = VK_SUBPASS_EXTERNAL,
            .srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask   = VK_ACCESS_SHADER_READ_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        };

        // ─[ Render Pass ]────────────────────────────────────────────────────
        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

        VkRenderPassCreateInfo renderPassInfo{
            .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .subpassCount    = 1,
            .pSubpasses      = &subpass,
            .dependencyCount = static_cast<uint32_t>(dependencies.size()),
            .pDependencies   = dependencies.data(),
        };

        VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &meshViewRenderPass));
    }

    // ╭─────────────────────────────────────────────────────────╮
    // │                     UI Render Pass                      │
    // ╰─────────────────────────────────────────────────────────╯
    {
        // ─[ Color Attachment ]───────────────────────────────────────────────
        VkAttachmentDescription colorAttachment{
            .format         = swapchainImageFormat,
            .samples        = VK_SAMPLE_COUNT_1_BIT,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        VkAttachmentReference colorAttachmentRef{
            .attachment = 0,
            .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        // ─[ Subpass ]────────────────────────────────────────────────────────
        VkSubpassDescription subpass{
            .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachmentRef,
        };

        VkSubpassDependency dependency{
            .srcSubpass    = VK_SUBPASS_EXTERNAL,
            .dstSubpass    = 0,
            .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        // ─[ Render Pass ]────────────────────────────────────────────────────
        VkRenderPassCreateInfo renderPassInfo{
            .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments    = &colorAttachment,
            .subpassCount    = 1,
            .pSubpasses      = &subpass,
            .dependencyCount = 1,
            .pDependencies   = &dependency,
        };

        VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &UIRenderPass));
    }
}

void Renderer::createGraphicsPipeline()
{
    // ─[ Shader Stage ]───────────────────────────────────────────────────
    VkShaderModule vertShaderModule = createShaderModule(SHADERS_DIR "/vert.spv");
    VkShaderModule fragShaderModule = createShaderModule(SHADERS_DIR "/frag.spv");

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertShaderModule,
        .pName  = "main",
    };

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragShaderModule,
        .pName  = "main",
    };

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // ─[ Vertex Input ]───────────────────────────────────────────────────
    auto bindingDescription    = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType                         = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions    = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions    = attributeDescriptions.data(),
    };

    // ─[ Input Assembly ]─────────────────────────────────────────────────
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    // ─[ Viewport and Scissor ]───────────────────────────────────────────
    VkPipelineViewportStateCreateInfo viewportInfo{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };

    // ─[ Dynamic States ]─────────────────────────────────────────────────
    std::vector<VkDynamicState> dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamicStateInfo{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };

    // ─[ Rasterization ]──────────────────────────────────────────────────
    VkPipelineRasterizationStateCreateInfo rasterizationInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable        = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode             = VK_POLYGON_MODE_FILL,
        .cullMode                = VK_CULL_MODE_BACK_BIT,
        .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable         = VK_FALSE,
        .lineWidth               = 1.f,
    };

    // ─[ Multisampling ]──────────────────────────────────────────────────
    VkPipelineMultisampleStateCreateInfo multisampleInfo{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable  = VK_FALSE,
    };

    // ─[ Depth and Stencil ]──────────────────────────────────────────────
    VkPipelineDepthStencilStateCreateInfo depthStencilInfo{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable       = VK_TRUE,
        .depthWriteEnable      = VK_TRUE,
        .depthCompareOp        = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable     = VK_FALSE,
    };

    // ─[ Color Blend ]────────────────────────────────────────────────────
    VkPipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo colorBlendInfo{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable   = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments    = &colorBlendAttachment,
    };

    // ─[ Pipeline Layout ]────────────────────────────────────────────────
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset     = 0,
        .size       = sizeof(PushConstantData),
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange,
    };

    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    // ─[ Pipeline ]───────────────────────────────────────────────────────
    VkGraphicsPipelineCreateInfo createInfo{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount          = 2,
        .pStages             = shaderStages,
        .pVertexInputState   = &vertexInputInfo,
        .pInputAssemblyState = &inputAssemblyInfo,
        .pViewportState      = &viewportInfo,
        .pRasterizationState = &rasterizationInfo,
        .pMultisampleState   = &multisampleInfo,
        .pDepthStencilState  = &depthStencilInfo,
        .pColorBlendState    = &colorBlendInfo,
        .pDynamicState       = &dynamicStateInfo,
        .layout              = pipelineLayout,
        .renderPass          = meshViewRenderPass,
        .subpass             = 0,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device,
                                       VK_NULL_HANDLE,
                                       1,
                                       &createInfo,
                                       nullptr,
                                       &graphicsPipeline));

    // ─[ Cleanup ]────────────────────────────────────────────────────────
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void Renderer::createCommandPools()
{
    // ─[ Graphics ]───────────────────────────────────────────────────────
    VkCommandPoolCreateInfo createInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
    };

    VK_CHECK(vkCreateCommandPool(device, &createInfo, nullptr, &graphicsCommandPool));
}

void Renderer::createSampler()
{
    VkSamplerCreateInfo samplerInfo{
        .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter               = VK_FILTER_LINEAR,
        .minFilter               = VK_FILTER_LINEAR,
        .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias              = 0.0f,
        .anisotropyEnable        = VK_FALSE,
        .compareEnable           = VK_FALSE,
        .compareOp               = VK_COMPARE_OP_ALWAYS,
        .minLod                  = 0.0f,
        .maxLod                  = 0.0f,
        .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &sampler));
}

void Renderer::initMeshManager()
{
    MeshManager::VulkanContext vulkanContext{
        .physicalDevice = physicalDevice,
        .device         = device,
        .commandPool    = graphicsCommandPool,
        .queue          = graphicsQueue,
    };

    meshManager.init(vulkanContext);
}

void Renderer::initEditorUI()
{
    ImGui_ImplVulkan_InitInfo initInfo{
        .ApiVersion         = VULKAN_API_VERSION,
        .Instance           = vkInstance,
        .PhysicalDevice     = physicalDevice,
        .Device             = device,
        .QueueFamily        = queueFamilyIndices.graphicsFamily.value(),
        .Queue              = graphicsQueue,
        .DescriptorPoolSize = 64, // TODO: calibrate pool size
        .MinImageCount      = static_cast<uint32_t>(swapchainImages.size()),
        .ImageCount         = static_cast<uint32_t>(swapchainImages.size()),
        .PipelineCache      = nullptr,
        .PipelineInfoMain{
            .RenderPass  = UIRenderPass,
            .Subpass     = 0,
            .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        },
        .Allocator = nullptr,
        .CheckVkResultFn =
            [](VkResult err) {
                if (err != VK_SUCCESS) {
                    throw std::runtime_error("ImGui internal Vulkan call failed with " +
                                             std::string(string_VkResult(err)));
                }
            },
    };

    editorUI.init(&initInfo, window.getNative());
}

void Renderer::createMeshViewResources()
{
    meshViewFrameData.extent = editorUI.getMeshViewportExtent();

    // ─[ Color Images ]───────────────────────────────────────────────────
    meshViewFrameData.colorImages.resize(maxFramesInFlight);
    meshViewFrameData.colorImagesMemory.resize(maxFramesInFlight);
    meshViewFrameData.colorImageViews.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        createImage(meshViewFrameData.extent.width,
                    meshViewFrameData.extent.height,
                    meshViewFrameData.format,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    meshViewFrameData.colorImages[i],
                    meshViewFrameData.colorImagesMemory[i]);

        createImageView(meshViewFrameData.colorImages[i],
                        meshViewFrameData.format,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        meshViewFrameData.colorImageViews[i]);
    }

    // ─[ Depth Image ]────────────────────────────────────────────────────
    VkFormat depthFormat = findDepthFormat();

    createImage(meshViewFrameData.extent.width,
                meshViewFrameData.extent.height,
                depthFormat,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                meshViewFrameData.depthImage,
                meshViewFrameData.depthImageMemory);

    createImageView(meshViewFrameData.depthImage,
                    depthFormat,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    meshViewFrameData.depthImageView);

    // ─[ Framebuffer ]────────────────────────────────────────────────────
    meshViewFrameData.framebuffers.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        std::array<VkImageView, 2> attachments = {meshViewFrameData.colorImageViews[i],
                                                  meshViewFrameData.depthImageView};

        VkFramebufferCreateInfo framebufferInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = meshViewRenderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .width           = meshViewFrameData.extent.width,
            .height          = meshViewFrameData.extent.height,
            .layers          = 1,
        };

        VK_CHECK(vkCreateFramebuffer(device,
                                     &framebufferInfo,
                                     nullptr,
                                     &meshViewFrameData.framebuffers[i]));
    }

    // ─[ Descriptor Sets ]────────────────────────────────────────────────
    meshViewFrameData.descriptorSets.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        meshViewFrameData.descriptorSets[i] =
            ImGui_ImplVulkan_AddTexture(sampler,
                                        meshViewFrameData.colorImageViews[i],
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void Renderer::createSwapchainFramebuffers()
{
    swapchainFramebuffers.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        std::array<VkImageView, 1> attachments = {
            swapchainImageViews[i],
        };

        VkFramebufferCreateInfo createInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = UIRenderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .width           = swapchainExtent.width,
            .height          = swapchainExtent.height,
            .layers          = 1,
        };

        VK_CHECK(vkCreateFramebuffer(device, &createInfo, nullptr, &swapchainFramebuffers[i]));
    }
}

void Renderer::allocateCommandBuffers()
{
    // ─[ Graphics ]───────────────────────────────────────────────────────
    graphicsCommandBuffers.resize(maxFramesInFlight);

    VkCommandBufferAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = graphicsCommandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<uint32_t>(graphicsCommandBuffers.size()),
    };

    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, graphicsCommandBuffers.data()));
}

void Renderer::createSyncObjects()
{
    imageAvailableSemaphores.resize(maxFramesInFlight);
    inFlightFences.resize(maxFramesInFlight);

    VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,

    };

    VkFenceCreateInfo fenceInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]));
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]));
    }
}

void Renderer::recordDrawCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIdx)
{
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    // ╭─────────────────────────────────────────────────────────╮
    // │              Pass 1: Off-screen Mesh View               │
    // ╰─────────────────────────────────────────────────────────╯

    if (meshViewFrameData.extent.width > 0 && meshViewFrameData.extent.height > 0) {
        // ─[ Begin Render Pass ]──────────────────────────────────────────────
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color        = {{0.f, 0.f, 0.f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassInfo{
            .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass  = meshViewRenderPass,
            .framebuffer = meshViewFrameData.framebuffers[currentFrame],
            .renderArea{
                .offset = {0, 0},
                .extent = meshViewFrameData.extent,
            },
            .clearValueCount = static_cast<uint32_t>(clearValues.size()),
            .pClearValues    = clearValues.data(),
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // ─[ Bind Resources ]─────────────────────────────────────────────────
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        meshManager.bindMeshResources(commandBuffer);

        // ─[ Dynamic States ]─────────────────────────────────────────────────
        VkViewport viewport{
            .x        = 0.f,
            .y        = 0.f,
            .width    = static_cast<float>(meshViewFrameData.extent.width),
            .height   = static_cast<float>(meshViewFrameData.extent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = meshViewFrameData.extent,
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // ─[ Push Constants ]─────────────────────────────────────────────────

        PushConstantData pushConstant = getPushConstantData();

        vkCmdPushConstants(commandBuffer,
                           pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT,
                           0,
                           sizeof(PushConstantData),
                           &pushConstant);

        // ─[ Draw ]───────────────────────────────────────────────────────────
        meshManager.drawMesh(commandBuffer);

        // ─[ End Render Pass ]────────────────────────────────────────────────
        vkCmdEndRenderPass(commandBuffer);
    }

    // ╭─────────────────────────────────────────────────────────╮
    // │                    Pass 2: Editor UI                    │
    // ╰─────────────────────────────────────────────────────────╯

    {
        // ─[ Begin Render Pass ]──────────────────────────────────────────────
        VkClearValue clearValue = {{{0.0f, 0.0f, 0.0f, 1.0f}}};

        VkRenderPassBeginInfo renderPassInfo{
            .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass  = UIRenderPass,
            .framebuffer = swapchainFramebuffers[imageIdx],
            .renderArea{
                .offset = {0, 0},
                .extent = swapchainExtent,
            },
            .clearValueCount = 1,
            .pClearValues    = &clearValue,
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // ─[ Draw ]───────────────────────────────────────────────────────────
        editorUI.draw(commandBuffer, meshViewFrameData.descriptorSets[currentFrame]);

        // ─[ End Render Pass ]────────────────────────────────────────────────
        vkCmdEndRenderPass(commandBuffer);
    }

    VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void Renderer::recreateMeshViewResources()
{
    if (meshViewFrameData.extent.width == 0 || meshViewFrameData.extent.height == 0) return;

    vkDeviceWaitIdle(device);

    cleanupMeshViewResources();
    createMeshViewResources();
}

void Renderer::cleanupMeshViewResources()
{
    // ─[ Color Images ]───────────────────────────────────────────────────
    for (size_t i = 0; i < maxFramesInFlight; i++) {
        vkDestroyFramebuffer(device, meshViewFrameData.framebuffers[i], nullptr);
        vkDestroyImageView(device, meshViewFrameData.colorImageViews[i], nullptr);
        vkDestroyImage(device, meshViewFrameData.colorImages[i], nullptr);
        vkFreeMemory(device, meshViewFrameData.colorImagesMemory[i], nullptr);
    }

    meshViewFrameData.framebuffers.clear();
    meshViewFrameData.colorImageViews.clear();
    meshViewFrameData.colorImages.clear();
    meshViewFrameData.colorImagesMemory.clear();

    // ─[ Depth Image ]────────────────────────────────────────────────────
    vkDestroyImageView(device, meshViewFrameData.depthImageView, nullptr);
    vkDestroyImage(device, meshViewFrameData.depthImage, nullptr);
    vkFreeMemory(device, meshViewFrameData.depthImageMemory, nullptr);

    // ─[ Descriptor Sets ]────────────────────────────────────────────────
    for (size_t i = 0; i < maxFramesInFlight; i++) {
        ImGui_ImplVulkan_RemoveTexture(meshViewFrameData.descriptorSets[i]);
    }

    meshViewFrameData.descriptorSets.clear();
}

void Renderer::recreateSwapchain()
{
    // Handle minimization
    int width = 0, height = 0;
    window.getFramebufferSize(&width, &height);
    while (width == 0 || height == 0) {
        window.getFramebufferSize(&width, &height);
        window.waitEvents();
    }

    vkDeviceWaitIdle(device);

    cleanupSwapchain();

    createSwapchain();
    createSwapchainImageViews();
    createSwapchainFramebuffers();
}

void Renderer::cleanupSwapchain()
{
    for (size_t i = 0; i < swapchainImages.size(); ++i) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);

        vkDestroyImageView(device, swapchainImageViews[i], nullptr);

        vkDestroyFramebuffer(device, swapchainFramebuffers[i], nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

// ╭─────────────────────────────────────────────────────────╮
// │                    HELPER FUNCTIONS                     │
// ╰─────────────────────────────────────────────────────────╯
bool Renderer::checkValidationLayersSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
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

std::vector<const char*> Renderer::getRequiredExtensions()
{
    uint32_t     glfwExtensionCount = 0;
    const char** glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if constexpr (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
Renderer::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
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

void Renderer::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
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
Renderer::CreateDebugUtilsMessengerEXT(VkInstance                                instance,
                                       const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                       const VkAllocationCallbacks*              pAllocator,
                                       VkDebugUtilsMessengerEXT*                 pDebugMessenger)
{
    auto func =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,
                                                                  "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr)
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void Renderer::DestroyDebugUtilsMessengerEXT(VkInstance                   instance,
                                             VkDebugUtilsMessengerEXT     debugMessenger,
                                             const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance,
        "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) func(instance, debugMessenger, pAllocator);
}

int Renderer::rateDeviceSuitability(VkPhysicalDevice dev)
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(dev, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(dev, &deviceFeatures);

    // ─[ Check Device Suitability ]───────────────────────────────────────
    bool isSuitable = findQueueFamilies(dev).isComplete() && checkDeviceExtensionSupport(dev) &&
                      querySwapchainSupport(dev).isAdequate();

    if (!isSuitable) return -1;

    // ─[ Rate Device ]───────────────────────────────────────────────────
    int score = 0;

    // Discrete GPU
    score += (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1000 : 0;

    return score;
}

QueueFamilyIndices Renderer::findQueueFamilies(VkPhysicalDevice dev)
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

bool Renderer::checkDeviceExtensionSupport(VkPhysicalDevice dev)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SwapchainSupportDetails Renderer::querySwapchainSupport(VkPhysicalDevice dev)
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

VkSurfaceFormatKHR
Renderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR
Renderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    return VK_PRESENT_MODE_FIFO_KHR;

    // for (const auto& availablePresentMode : availablePresentModes) {
    //     if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
    //         return availablePresentMode;
    //     }
    // }
    //
    // return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Renderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    // Special value to indicate that the extent should be chosen and set manually
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    int width, height;
    window.getFramebufferSize(&width, &height);

    VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    actualExtent.width = std::clamp(actualExtent.width,
                                    capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);

    actualExtent.height = std::clamp(actualExtent.height,
                                     capabilities.minImageExtent.height,
                                     capabilities.maxImageExtent.height);

    return actualExtent;
}

VkShaderModule Renderer::createShaderModule(const char* bytecodePath)
{
    // ─[ Read File ]──────────────────────────────────────────────────────
    std::ifstream file(bytecodePath, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error(std::format("Failed to open file: {}", bytecodePath));

    size_t            fileSize = (size_t)file.tellg();
    std::vector<char> codeBuffer(fileSize);

    file.seekg(0);
    file.read(codeBuffer.data(), fileSize);

    file.close();

    // ─[ Create Shader Module ]───────────────────────────────────────────
    VkShaderModuleCreateInfo createInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = codeBuffer.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(codeBuffer.data()),
    };

    VkShaderModule shaderModule;

    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
}

VkFormat Renderer::findSupportedFormat(const std::vector<VkFormat>& candidates,
                                       VkImageTiling                tiling,
                                       VkFormatFeatureFlags         features)
{
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                 (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format!");
}

VkFormat Renderer::findDepthFormat()
{
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void Renderer::createImage(uint32_t              width,
                           uint32_t              height,
                           VkFormat              format,
                           VkImageTiling         tiling,
                           VkImageUsageFlags     usage,
                           VkMemoryPropertyFlags properties,
                           VkImage&              image,
                           VkDeviceMemory&       imageMemory)
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

    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &image));

    // ─[ Allocate Image Memory ]──────────────────────────────────────────
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, image, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{
        .sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex =
            utils::findMemoryType(memoryRequirements.memoryTypeBits, properties, physicalDevice),
    };

    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory));

    VK_CHECK(vkBindImageMemory(device, image, imageMemory, 0));
}

void Renderer::createImageView(VkImage            image,
                               VkFormat           format,
                               VkImageAspectFlags aspectFlags,
                               VkImageView&       imageView)
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

    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &imageView));
}

void Renderer::updateCamera()
{
    auto dragOffset   = window.consumeDragOffset();
    auto scrollOffset = window.consumeScrollOffset();

    if (editorUI.isMeshViewportFocused()) camera.orbit(dragOffset);
    if (editorUI.isMeshViewportHovered()) camera.adjustRadius(scrollOffset);
}

PushConstantData Renderer::getPushConstantData()
{
    updateCamera();

    glm::mat4 proj =
        camera.getProjMat((float)meshViewFrameData.extent.width / meshViewFrameData.extent.height);
    proj[1][1] *= -1.f;

    return PushConstantData{.viewProjMat = proj * camera.getViewMat()};
}
