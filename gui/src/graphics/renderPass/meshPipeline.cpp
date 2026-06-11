#include "meshPipeline.hpp"
#include <sstream>
#include <fstream>

void MeshPipeline::init(const VulkanContext& vkCtx, VkRenderPass renderPass)
{
    // ─[ Shader Stage ]───────────────────────────────────────────────────
    VkShaderModule vertShaderModule = createShaderModule(vkCtx, SHADERS_DIR "/vert.spv");
    VkShaderModule fragShaderModule = createShaderModule(vkCtx, SHADERS_DIR "/frag.spv");

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
        VK_DYNAMIC_STATE_POLYGON_MODE_EXT,
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
        .polygonMode             = VK_POLYGON_MODE_LINE, // Dynamic
        // TODO: Enable backface culling after setting up mush mesh triangle orientations
        .cullMode        = VK_CULL_MODE_NONE,
        .frontFace       = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth       = 1.f,
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

    VK_CHECK(vkCreatePipelineLayout(vkCtx.device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

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
        .renderPass          = renderPass,
        .subpass             = 0,
    };

    VK_CHECK(vkCreateGraphicsPipelines(vkCtx.device,
                                       VK_NULL_HANDLE,
                                       1,
                                       &createInfo,
                                       nullptr,
                                       &pipeline));

    // ─[ Cleanup ]────────────────────────────────────────────────────────
    vkDestroyShaderModule(vkCtx.device, vertShaderModule, nullptr);
    vkDestroyShaderModule(vkCtx.device, fragShaderModule, nullptr);
}

void MeshPipeline::shutdown(const VulkanContext& vkCtx)
{
    vkDestroyPipeline(vkCtx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(vkCtx.device, pipelineLayout, nullptr);
}

VkPipeline       MeshPipeline::getGraphicsPipeline() const { return pipeline; }
VkPipelineLayout MeshPipeline::getGraphicsPipelineLayout() const { return pipelineLayout; }

VkShaderModule MeshPipeline::createShaderModule(const VulkanContext& vkCtx,
                                                const char*          bytecodePath)
{
    // ─[ Read File ]──────────────────────────────────────────────────────
    std::ifstream file(bytecodePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::stringstream ss;
        ss << "Failed to open file: " << bytecodePath;
        throw std::runtime_error(ss.str());
    }

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

    VK_CHECK(vkCreateShaderModule(vkCtx.device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
}
