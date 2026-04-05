#include "meshManager.hpp"

void MeshManager::init(VulkanContext vulkanContext)
{
    this->vulkanContext = vulkanContext;
    gmsh::initialize();
#ifdef NDEBUG
    gmsh::option::setNumber("General.Verbosity", 0);
#else
    gmsh::option::setNumber("General.Verbosity", 5);
#endif

    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f, 0.5f}, {0.75f, 0.25f, 0.25f}}, // 0
        {{0.5f, -0.5f, 0.5f}, {0.75f, 0.25f, 0.25f}},  // 1
        {{0.5f, 0.5f, 0.5f}, {0.75f, 0.25f, 0.25f}},   // 2
        {{-0.5f, 0.5f, 0.5f}, {0.75f, 0.25f, 0.25f}},  // 3

        {{0.5f, -0.5f, -0.5f}, {0.25f, 0.70f, 0.25f}},  // 4
        {{-0.5f, -0.5f, -0.5f}, {0.25f, 0.70f, 0.25f}}, // 5
        {{-0.5f, 0.5f, -0.5f}, {0.25f, 0.70f, 0.25f}},  // 6
        {{0.5f, 0.5f, -0.5f}, {0.25f, 0.70f, 0.25f}},   // 7

        {{-0.5f, 0.5f, 0.5f}, {0.25f, 0.28f, 0.75f}},  // 8
        {{0.5f, 0.5f, 0.5f}, {0.25f, 0.28f, 0.75f}},   // 9
        {{0.5f, 0.5f, -0.5f}, {0.25f, 0.28f, 0.75f}},  // 10
        {{-0.5f, 0.5f, -0.5f}, {0.25f, 0.28f, 0.75f}}, // 11

        {{-0.5f, -0.5f, -0.5f}, {0.50f, 0.58f, 0.22f}}, // 12
        {{0.5f, -0.5f, -0.5f}, {0.50f, 0.58f, 0.22f}},  // 13
        {{0.5f, -0.5f, 0.5f}, {0.50f, 0.58f, 0.22f}},   // 14
        {{-0.5f, -0.5f, 0.5f}, {0.50f, 0.58f, 0.22f}},  // 15

        {{0.5f, -0.5f, 0.5f}, {0.58f, 0.24f, 0.52f}},  // 16
        {{0.5f, -0.5f, -0.5f}, {0.58f, 0.24f, 0.52f}}, // 17
        {{0.5f, 0.5f, -0.5f}, {0.58f, 0.24f, 0.52f}},  // 18
        {{0.5f, 0.5f, 0.5f}, {0.58f, 0.24f, 0.52f}},   // 19

        {{-0.5f, -0.5f, -0.5f}, {0.22f, 0.50f, 0.50f}}, // 20
        {{-0.5f, -0.5f, 0.5f}, {0.22f, 0.50f, 0.50f}},  // 21
        {{-0.5f, 0.5f, 0.5f}, {0.22f, 0.50f, 0.50f}},   // 22
        {{-0.5f, 0.5f, -0.5f}, {0.22f, 0.50f, 0.50f}}   // 23
    };

    const std::vector<uint32_t> indices = {
        0,  1,  2,  2,  3,  0,  // Front
        4,  5,  6,  6,  7,  4,  // Back
        8,  9,  10, 10, 11, 8,  // Top
        12, 13, 14, 14, 15, 12, // Bottom
        16, 17, 18, 18, 19, 16, // Right
        20, 21, 22, 22, 23, 20  // Left
    };

    currentMeshData.indexCount = static_cast<uint32_t>(indices.size());
    createVertexBuffer(vertices);
    createIndexBuffer(indices);
}
void MeshManager::cleanup()
{
    cleanupCurrentMesh();
    gmsh::finalize();
}

void MeshManager::loadMesh(std::string filePath)
{
    // ─[ Cleanup ]────────────────────────────────────────────────────────
    cleanupCurrentMesh();

    // ─[ Load Mesh ]──────────────────────────────────────────────────────
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;

    gmsh::open(filePath);

    std::vector<std::size_t> nodeTags;
    std::vector<double>      coords, parametricCoords;
    gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords);

    std::unordered_map<std::size_t, uint32_t> tagToIndex;
    for (std::size_t i = 0; i < nodeTags.size(); i++) {
        tagToIndex[nodeTags[i]] = static_cast<uint32_t>(vertices.size());
        vertices.push_back(
            {{(float)coords[3 * i], (float)coords[3 * i + 1], (float)coords[3 * i + 2]},
             {0.6f, 0.6f, 0.6f}});
    }

    std::vector<int>                      elemTypes;
    std::vector<std::vector<std::size_t>> elemTags, elemNodeTags;
    gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, 2);

    for (std::size_t t = 0; t < elemTypes.size(); t++) {
        std::string         name;
        int                 dim, order, numNodes, numPrimaryNodes;
        std::vector<double> localNodeCoord;
        gmsh::model::mesh::getElementProperties(elemTypes[t],
                                                name,
                                                dim,
                                                order,
                                                numNodes,
                                                localNodeCoord,
                                                numPrimaryNodes);

        const auto& nodeList = elemNodeTags[t];
        size_t      numElems = nodeList.size() / numNodes;

        for (size_t e = 0; e < numElems; e++) {
            if (numNodes == 3) {
                for (int n = 0; n < 3; n++)
                    indices.push_back(tagToIndex.at(nodeList[e * 3 + n]));
            }
            else if (numNodes == 4) {
                uint32_t i0 = tagToIndex.at(nodeList[e * 4 + 0]);
                uint32_t i1 = tagToIndex.at(nodeList[e * 4 + 1]);
                uint32_t i2 = tagToIndex.at(nodeList[e * 4 + 2]);
                uint32_t i3 = tagToIndex.at(nodeList[e * 4 + 3]);
                indices.insert(indices.end(), {i0, i1, i2, i0, i2, i3});
            }
        }
    }

    // ─[ Create Mesh Resources ]──────────────────────────────────────────
    currentMeshData.indexCount = static_cast<uint32_t>(indices.size());
    createVertexBuffer(vertices);
    createIndexBuffer(indices);
}

void MeshManager::bindMeshResources(VkCommandBuffer commandBuffer)
{
    VkBuffer     vertexBuffers[] = {currentMeshData.vertexBuffer};
    VkDeviceSize offsets[]       = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(commandBuffer, currentMeshData.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
}

void MeshManager::drawMesh(VkCommandBuffer commandBuffer)
{

    vkCmdDrawIndexed(commandBuffer, currentMeshData.indexCount, 1, 0, 0, 0);
}

void MeshManager::cleanupCurrentMesh()
{
    vkDeviceWaitIdle(vulkanContext.device);

    vkDestroyBuffer(vulkanContext.device, currentMeshData.vertexBuffer, nullptr);
    vkFreeMemory(vulkanContext.device, currentMeshData.vertexBufferMemory, nullptr);

    vkDestroyBuffer(vulkanContext.device, currentMeshData.indexBuffer, nullptr);
    vkFreeMemory(vulkanContext.device, currentMeshData.indexBufferMemory, nullptr);

    currentMeshData.vertexBuffer       = VK_NULL_HANDLE;
    currentMeshData.vertexBufferMemory = VK_NULL_HANDLE;
    currentMeshData.indexBuffer        = VK_NULL_HANDLE;
    currentMeshData.indexBufferMemory  = VK_NULL_HANDLE;
    currentMeshData.indexCount         = 0;
}

void MeshManager::createVertexBuffer(const std::vector<Vertex>& vertices)
{
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    // ─[ Staging Buffer ]─────────────────────────────────────────────────
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stagingBuffer,
                        stagingBufferMemory,
                        vulkanContext.physicalDevice,
                        vulkanContext.device);

    // Copy data
    void* data;
    VK_CHECK(vkMapMemory(vulkanContext.device, stagingBufferMemory, 0, bufferSize, 0, &data));
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkanContext.device, stagingBufferMemory);

    // ─[ Vertex Buffer ]──────────────────────────────────────────────────
    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        currentMeshData.vertexBuffer,
                        currentMeshData.vertexBufferMemory,
                        vulkanContext.physicalDevice,
                        vulkanContext.device);

    utils::copyBuffer(stagingBuffer,
                      currentMeshData.vertexBuffer,
                      bufferSize,
                      vulkanContext.device,
                      vulkanContext.commandPool,
                      vulkanContext.queue);

    // ─[ Cleanup ]────────────────────────────────────────────────────────
    vkDestroyBuffer(vulkanContext.device, stagingBuffer, nullptr);
    vkFreeMemory(vulkanContext.device, stagingBufferMemory, nullptr);
}

void MeshManager::createIndexBuffer(const std::vector<uint32_t>& indices)
{
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    // ─[ Staging Buffer ]─────────────────────────────────────────────────
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stagingBuffer,
                        stagingBufferMemory,
                        vulkanContext.physicalDevice,
                        vulkanContext.device);

    // Copy data
    void* data;
    VK_CHECK(vkMapMemory(vulkanContext.device, stagingBufferMemory, 0, bufferSize, 0, &data));
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkanContext.device, stagingBufferMemory);

    // ─[ Index Buffer ]───────────────────────────────────────────────────
    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        currentMeshData.indexBuffer,
                        currentMeshData.indexBufferMemory,
                        vulkanContext.physicalDevice,
                        vulkanContext.device);

    utils::copyBuffer(stagingBuffer,
                      currentMeshData.indexBuffer,
                      bufferSize,
                      vulkanContext.device,
                      vulkanContext.commandPool,
                      vulkanContext.queue);

    // ─[ Cleanup ]────────────────────────────────────────────────────────
    vkDestroyBuffer(vulkanContext.device, stagingBuffer, nullptr);
    vkFreeMemory(vulkanContext.device, stagingBufferMemory, nullptr);
}
