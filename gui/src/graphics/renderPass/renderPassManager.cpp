#include "renderPassManager.hpp"
#include "../../common/config.hpp"

void RenderPassManager::init(const VulkanContext& vkCtx, GLFWwindow* glfwWindow)
{
    createRenderPasses(vkCtx, meshViewResources.resolveFormats(vkCtx));
    meshViewResources.init(vkCtx, renderPasses[MESH_VIEW]);
}

void RenderPassManager::shutdown(const VulkanContext& vkCtx)
{
    meshViewResources.shutdown(vkCtx);
    vkDestroyRenderPass(vkCtx.device, renderPasses[MESH_VIEW], nullptr);
    vkDestroyRenderPass(vkCtx.device, renderPasses[UI], nullptr);
}

void RenderPassManager::initMeshViewTextureDesc() { meshViewResources.initTextureDesc(); }
void RenderPassManager::clearMeshViewTextureDesc() { meshViewResources.clearTextureDesc(); }

void RenderPassManager::update(const VulkanContext& vkCtx, const UIState& uiState)
{
    VkExtent2D uiMeshViewExtent = {.width  = (uint32_t)uiState.meshView.size.x,
                                   .height = (uint32_t)uiState.meshView.size.y};

    if (uiMeshViewExtent.width == 0 || uiMeshViewExtent.height == 0) return;

    VkExtent2D currMeshViewExtent = meshViewResources.getImageExtent();

    if (uiMeshViewExtent.width != currMeshViewExtent.width ||
        uiMeshViewExtent.height != currMeshViewExtent.height) {

        meshViewResources.reset(vkCtx, renderPasses[MESH_VIEW], uiMeshViewExtent);
    }
}

VkRenderPass RenderPassManager::getRenderPass(RenderPassEnum type) { return renderPasses.at(type); }

ImGui_ImplVulkan_PipelineInfo RenderPassManager::getImGuiPipelineInfo() const
{
    return ImGui_ImplVulkan_PipelineInfo{
        .RenderPass  = renderPasses.at(UI),
        .Subpass     = 0,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
    };
}

VkDescriptorSet RenderPassManager::getMeshViewTextureDesc(uint32_t idx) const
{
    return meshViewResources.getMeshViewTextureDesc(idx);
}

void RenderPassManager::createRenderPasses(const VulkanContext& vkCtx,
                                           AttachmentFormats    meshViewFormats)
{
    // ╭─────────────────────────────────────────────────────────╮
    // │                  Mesh View Render Pass                  │
    // ╰─────────────────────────────────────────────────────────╯
    {
        // ─[ Color Attachment ]───────────────────────────────────────────────
        VkAttachmentDescription colorAttachment{
            .format         = meshViewFormats.color,
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
            .format         = meshViewFormats.depth,
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

        VK_CHECK(
            vkCreateRenderPass(vkCtx.device, &renderPassInfo, nullptr, &renderPasses[MESH_VIEW]));
    }

    // ╭─────────────────────────────────────────────────────────╮
    // │                     UI Render Pass                      │
    // ╰─────────────────────────────────────────────────────────╯
    {
        // ─[ Color Attachment ]───────────────────────────────────────────────
        VkAttachmentDescription colorAttachment{
            .format         = vkCtx.swapchainImageFormat,
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

        VK_CHECK(vkCreateRenderPass(vkCtx.device, &renderPassInfo, nullptr, &renderPasses[UI]));
    }
}

void RenderPassManager::recordDrawCommandBuffer(const VulkanContext& vkCtx,
                                                VkCommandBuffer      commandBuffer,
                                                VkFramebuffer        swapchainFramebuffer,
                                                VkExtent2D           swapchainExtent,
                                                uint32_t             frameIdx,
                                                uint32_t             imageIdx,
                                                PushConstantData     pushConstant,
                                                const UIState&       uiState,
                                                const MeshData&      meshData)
{
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    // ╭─────────────────────────────────────────────────────────╮
    // │              Pass 1: Off-screen Mesh View               │
    // ╰─────────────────────────────────────────────────────────╯

    if (uiState.meshView.size.x > 0.f && uiState.meshView.size.y > 0.f) {
        VkExtent2D meshViewExtent = meshViewResources.getImageExtent();

        // ─[ Begin Render Pass ]──────────────────────────────────────────────
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color        = Config::render.clearColor;
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassInfo{
            .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass  = renderPasses[MESH_VIEW],
            .framebuffer = meshViewResources.getFrameBuffer(frameIdx),
            .renderArea{
                .offset = {0, 0},
                .extent = meshViewExtent,
            },
            .clearValueCount = static_cast<uint32_t>(clearValues.size()),
            .pClearValues    = clearValues.data(),
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // ─[ Bind Resources ]─────────────────────────────────────────────────
        vkCmdBindPipeline(commandBuffer,
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          meshViewResources.getGraphicsPipeline());

        VkBuffer     vertexBuffers[] = {meshData.vertexBuffer};
        VkDeviceSize offsets[]       = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, meshData.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // ─[ Dynamic States ]─────────────────────────────────────────────────
        VkViewport viewport{
            .x        = 0.f,
            .y        = 0.f,
            .width    = (float)meshViewExtent.width,
            .height   = (float)meshViewExtent.height,
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = meshViewExtent,
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCtx.vkCmdSetPolygonModeEXT(commandBuffer, uiState.meshView.polygoneMode);

        // ─[ Push Constants ]─────────────────────────────────────────────────
        vkCmdPushConstants(commandBuffer,
                           meshViewResources.getGraphicsPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT,
                           0,
                           sizeof(PushConstantData),
                           &pushConstant);

        // ─[ Draw ]───────────────────────────────────────────────────────────
        vkCmdDrawIndexed(commandBuffer, meshData.indexCount, 1, 0, 0, 0);

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
            .renderPass  = renderPasses[UI],
            .framebuffer = swapchainFramebuffer,
            .renderArea{
                .offset = {0, 0},
                .extent = swapchainExtent,
            },
            .clearValueCount = 1,
            .pClearValues    = &clearValue,
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // ─[ Draw ]───────────────────────────────────────────────────────────
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

        // ─[ End Render Pass ]────────────────────────────────────────────────
        vkCmdEndRenderPass(commandBuffer);
    }

    VK_CHECK(vkEndCommandBuffer(commandBuffer));
}
