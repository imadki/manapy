#include "./meshManager.hpp"
#include "../common/config.hpp"
#include <cstring>

void MeshManager::init(const VulkanContext& vkCtx)
{
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

    meshData.indexCount = static_cast<uint32_t>(indices.size());
    createVertexBuffer(vkCtx, vertices, &meshData.vertexBuffer, &meshData.vertexBufferMemory);
    createIndexBuffer(vkCtx, indices, &meshData.indexBuffer, &meshData.indexBufferMemory);
    meshData.modelMatrix = buildMeshModelMatrix(vertices);
}
void MeshManager::shutdown(const VulkanContext& vkCtx)
{
    meshData.clear(vkCtx);
    gmsh::finalize();
}

void MeshManager::update(const VulkanContext& vkCtx, const UIState& uiState)
{
    if (uiState.meshFile.isSelected) {
        loadMesh(vkCtx, uiState.meshFile.path);
    }
}

const MeshData& MeshManager::getMeshData() const { return meshData; }

void MeshManager::loadMesh(const VulkanContext& vkCtx, std::string filePath)
{
    // ─[ Cleanup ]────────────────────────────────────────────────────────
    meshData.clear(vkCtx);

    // ─[ Load Mesh ]──────────────────────────────────────────────────────
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;

    gmsh::open(filePath);

    // ─[ 1. Cache Global Node Coordinates ]───────────────────────────────
    // We get all nodes globally just to know where they are in 3D space.
    std::vector<std::size_t> nodeTags;
    std::vector<double>      coords, parametricCoords;
    gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords);

    // Map global node tags to their (x,y,z) coords for fast O(1) lookup
    std::unordered_map<std::size_t, std::array<double, 3>> globalNodeCoords;
    for (std::size_t i = 0; i < nodeTags.size(); i++) {
        globalNodeCoords[nodeTags[i]] = {coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]};
    }

    // ─[ 2. Map Physical Groups to Colors ]───────────────────────────────
    std::vector<std::pair<int, int>> physicalGroups;
    gmsh::model::getPhysicalGroups(physicalGroups, 2);

    // A rich, vibrant palette with higher saturation for clear 3D visibility
    const std::vector<std::array<float, 3>> palette = {
        {0.80f, 0.25f, 0.30f}, // Red
        {0.15f, 0.45f, 0.75f}, // Blue
        {0.20f, 0.65f, 0.35f}, // Green
        {0.85f, 0.55f, 0.15f}, // Gold
        {0.55f, 0.30f, 0.70f}, // Purple
        {0.15f, 0.65f, 0.60f}, // Teal
        {0.80f, 0.40f, 0.15f}  // Orange
    };

    std::unordered_map<int, std::array<float, 3>> groupColors;
    for (size_t i = 0; i < physicalGroups.size(); i++) {
        int tag = physicalGroups[i].second;

        // Assign colors sequentially from the palette, wrapping around if
        // you have more physical groups than available colors.
        groupColors[tag] = palette[i % palette.size()];
    }

    // ─[ 3. Process Entities & Duplicate Boundary Vertices ]──────────────
    std::vector<std::pair<int, int>> entities;
    gmsh::model::getEntities(entities, 2);

    for (const auto& entity : entities) {
        int entityTag = entity.second;

        // Determine this entity's color
        std::vector<int> physicalTags;
        gmsh::model::getPhysicalGroupsForEntity(2, entityTag, physicalTags);
        std::array<float, 3> entityColor = {0.5f, 0.5f, 0.5f};
        if (!physicalTags.empty()) {
            entityColor = groupColors[physicalTags[0]];
        }

        // LOCAL map: Ensures vertices are unique to THIS entity.
        // Boundary nodes will be re-added as new vertices by neighboring entities.
        std::unordered_map<std::size_t, uint32_t> localTagToIndex;

        std::vector<int>                      elemTypes;
        std::vector<std::vector<std::size_t>> elemTags, elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, 2, entityTag);

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
                std::vector<uint32_t> currentElemIndices;

                // Process each node in the current element (triangle or quad)
                for (int n = 0; n < numNodes; n++) {
                    std::size_t nodeTag = nodeList[e * numNodes + n];

                    // If we haven't created a vertex for this node IN THIS ENTITY yet, create it.
                    if (localTagToIndex.find(nodeTag) == localTagToIndex.end()) {
                        const auto& c = globalNodeCoords[nodeTag];

                        uint32_t newVertexIndex  = static_cast<uint32_t>(vertices.size());
                        localTagToIndex[nodeTag] = newVertexIndex;

                        vertices.push_back(
                            {.position = {(float)c[0], (float)c[1], (float)c[2]},
                             .color    = {entityColor[0], entityColor[1], entityColor[2]}});
                    }

                    // Add the local index to our temporary list for this element
                    currentElemIndices.push_back(localTagToIndex[nodeTag]);
                }

                // Push the actual indices to the Vulkan index buffer
                if (numNodes == 3) {
                    indices.push_back(currentElemIndices[0]);
                    indices.push_back(currentElemIndices[1]);
                    indices.push_back(currentElemIndices[2]);
                }
                else if (numNodes == 4) {
                    indices.insert(indices.end(),
                                   {currentElemIndices[0],
                                    currentElemIndices[1],
                                    currentElemIndices[2],
                                    currentElemIndices[0],
                                    currentElemIndices[2],
                                    currentElemIndices[3]});
                }
            }
        }
    }

    // ─[ Create Mesh Resources ]──────────────────────────────────────────
    meshData.indexCount = static_cast<uint32_t>(indices.size());
    createVertexBuffer(vkCtx, vertices, &meshData.vertexBuffer, &meshData.vertexBufferMemory);
    createIndexBuffer(vkCtx, indices, &meshData.indexBuffer, &meshData.indexBufferMemory);
    meshData.modelMatrix = buildMeshModelMatrix(vertices);
}

void MeshManager::createVertexBuffer(const VulkanContext&       vkCtx,
                                     const std::vector<Vertex>& vertices,
                                     VkBuffer*                  vertexBuffer,
                                     VkDeviceMemory*            vertexBufferMemory)
{
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    // ─[ Staging Buffer ]─────────────────────────────────────────────────
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &stagingBuffer,
                        &stagingBufferMemory,
                        vkCtx.physicalDevice,
                        vkCtx.device);

    // Copy data
    void* data;
    VK_CHECK(vkMapMemory(vkCtx.device, stagingBufferMemory, 0, bufferSize, 0, &data));
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(vkCtx.device, stagingBufferMemory);

    // ─[ Vertex Buffer ]──────────────────────────────────────────────────
    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        vertexBuffer,
                        vertexBufferMemory,
                        vkCtx.physicalDevice,
                        vkCtx.device);

    utils::copyBuffer(stagingBuffer,
                      *vertexBuffer,
                      bufferSize,
                      vkCtx.device,
                      vkCtx.graphicsCommandPool,
                      vkCtx.graphicsQueue);

    // ─[ Cleanup ]────────────────────────────────────────────────────────
    vkDestroyBuffer(vkCtx.device, stagingBuffer, nullptr);
    vkFreeMemory(vkCtx.device, stagingBufferMemory, nullptr);
}

void MeshManager::createIndexBuffer(const VulkanContext&         vkCtx,
                                    const std::vector<uint32_t>& indices,
                                    VkBuffer*                    indexBuffer,
                                    VkDeviceMemory*              indexBufferMemory)
{
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    // ─[ Staging Buffer ]─────────────────────────────────────────────────
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &stagingBuffer,
                        &stagingBufferMemory,
                        vkCtx.physicalDevice,
                        vkCtx.device);

    // Copy data
    void* data;
    VK_CHECK(vkMapMemory(vkCtx.device, stagingBufferMemory, 0, bufferSize, 0, &data));
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(vkCtx.device, stagingBufferMemory);

    // ─[ Index Buffer ]───────────────────────────────────────────────────
    utils::createBuffer(bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        indexBuffer,
                        indexBufferMemory,
                        vkCtx.physicalDevice,
                        vkCtx.device);

    utils::copyBuffer(stagingBuffer,
                      *indexBuffer,
                      bufferSize,
                      vkCtx.device,
                      vkCtx.graphicsCommandPool,
                      vkCtx.graphicsQueue);

    // ─[ Cleanup ]────────────────────────────────────────────────────────
    vkDestroyBuffer(vkCtx.device, stagingBuffer, nullptr);
    vkFreeMemory(vkCtx.device, stagingBufferMemory, nullptr);
}

glm::mat4 MeshManager::buildMeshModelMatrix(const std::vector<Vertex>& vertices)
{
    // ─[ Center and Scale Mesh ]──────────────────────────────────────────
    if (vertices.empty()) return glm::mat4(1.f);
    // WARNING: Should only uses vertices of 2D primitives of the mesh

    // ─[ Get Bounding Box ]───────────────────────────────────────────────
    glm::vec3 boundingBox[2] = {{
                                    vertices[0].position.x,
                                    vertices[0].position.y,
                                    vertices[0].position.z,
                                },
                                {
                                    vertices[0].position.x,
                                    vertices[0].position.y,
                                    vertices[0].position.z,
                                }};

    for (const auto vertex : vertices) {
        boundingBox[0] = glm::min(boundingBox[0], vertex.position);
        boundingBox[1] = glm::max(boundingBox[1], vertex.position);
    }

    glm::vec3 size     = boundingBox[1] - boundingBox[0];
    float scaleFactor  = Config::mesh.defaultMeshSize / glm::max(size.x, glm::max(size.y, size.z));
    glm::mat4 scaleMat = glm::scale(glm::mat4(1.f), scaleFactor * glm::vec3(1.f));

    glm::vec3 center = Config::mesh.worldMeshAnchor + (boundingBox[0] + boundingBox[1]) / 2.f;

    return glm::translate(scaleMat, -center);
}
