#pragma once

#include <glm/glm.hpp>

struct CameraData {
    glm::mat4 projectionMatrix = glm::mat4{1.f};
    glm::mat4 viewMatrix       = glm::mat4{1.f};
};
