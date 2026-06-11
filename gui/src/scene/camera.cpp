#include <glm/gtc/matrix_transform.hpp>
#include "./camera.hpp"
#include "../common/config.hpp"

Camera::Camera()
    : position(Config::mesh.worldMeshAnchor + Config::mesh.defaultMeshSize * glm::vec3(1.0f)),
      fov(60.f), near(.01f), far(100.f), dragSensitivity(.8f, .6f), scrollSensitivity(.5f)
{
    glm::vec3 offset = position - Config::mesh.worldMeshAnchor;

    radius    = glm::length(offset);
    azimuthal = atan2(position.y, position.x);
    polar     = acos(position.z / radius);
}

void Camera::update(const InputState& inputState, const UIState& uiState)
{
    if (inputState.mouseLeftDown && uiState.meshView.isFocused) {
        orbit(inputState.mouseDelta);
    }

    if (uiState.meshView.isHovered) {
        adjustRadius(inputState.scrollDelta);
    }

    data.viewMatrix = glm::lookAt(position, Config::mesh.worldMeshAnchor, Config::mesh.worldUp);

    if (uiState.meshView.size.x > 0.f && uiState.meshView.size.y > 0.f) {
        float aspectRatio     = uiState.meshView.size.x / uiState.meshView.size.y;
        data.projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, near, far);
    }
}

const CameraData& Camera::getCameraData() const { return data; }

void Camera::orbit(glm::vec2 offset)
{
    const static float minPolar = glm::radians(1.f);
    const static float maxPolar = glm::radians(179.f);

    if (!glm::length(offset)) return;

    azimuthal += glm::radians(-offset.x * dragSensitivity.x);

    polar += glm::radians(-offset.y * dragSensitivity.y);
    polar = glm::clamp(polar, minPolar, maxPolar);

    updatePosition();
};

void Camera::adjustRadius(float offset)
{
    const static float minRadius = 2 * near;
    const static float maxRadius = far - Config::mesh.defaultMeshSize;

    const static float controlledZoomRadius = 10.f;

    float controlledZoomFactor = fmin(radius, controlledZoomRadius) / controlledZoomRadius;

    radius += -offset * scrollSensitivity * controlledZoomFactor;
    radius = glm::clamp(radius, minRadius, maxRadius);

    updatePosition();
}

void Camera::updatePosition()
{
    position.x = Config::mesh.worldMeshAnchor.x + radius * sin(polar) * cos(azimuthal);
    position.y = Config::mesh.worldMeshAnchor.y + radius * sin(polar) * sin(azimuthal);
    position.z = Config::mesh.worldMeshAnchor.z + radius * cos(polar);
}
