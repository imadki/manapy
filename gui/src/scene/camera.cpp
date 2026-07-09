#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include "./camera.hpp"
#include "../common/config.hpp"
#include "glm/geometric.hpp"

Camera::Camera() : pivot(0.f, 0.f, 0.f)
{
    position = pivot + Config::mesh.defaultMeshSize * glm::vec3(1.0f);

    glm::vec3 offset = position - pivot;

    radius    = glm::length(offset);
    azimuthal = atan2(position.y, position.x);
    polar     = acos(position.z / radius);
}

void Camera::update(const InputState& inputState, const UIState& uiState)
{
    this->fov          = uiState.meshView.cameraFov;
    this->orbitSpeed.x = uiState.meshView.orbitSpeed.x;
    this->orbitSpeed.y = uiState.meshView.orbitSpeed.y;
    this->zoomSpeed    = uiState.meshView.zoomSpeed;

    if (inputState.mouseMiddleDown && uiState.meshView.isFocused)
        adjustPivot(inputState.mouseDelta);
    if (inputState.mouseLeftDown && uiState.meshView.isFocused) orbit(inputState.mouseDelta);
    if (uiState.meshView.isHovered) adjustRadius(inputState.scrollDelta);

    // ─[ Update Internal State ]──────────────────────────────────────────
    data.viewMatrix = glm::lookAt(position, pivot, Config::mesh.worldUp);

    if (uiState.meshView.size.x > 0.f && uiState.meshView.size.y > 0.f) {
        float aspectRatio     = uiState.meshView.size.x / uiState.meshView.size.y;
        data.projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, near, far);
    }
}

const CameraData& Camera::getCameraData() const { return data; }

void Camera::orbit(const glm::vec2& offset)
{
    const static float minPolar = glm::radians(1.f);
    const static float maxPolar = glm::radians(179.f);

    if (!glm::length(offset)) return;

    azimuthal += glm::radians(-offset.x * orbitSpeed.x);

    polar += glm::radians(-offset.y * orbitSpeed.y);
    polar = glm::clamp(polar, minPolar, maxPolar);

    updatePosition();
};

void Camera::adjustRadius(float offset)
{
    const static float minRadius = 2 * near;
    const static float maxRadius = far - Config::mesh.defaultMeshSize;

    const static float controlledZoomRadius = 10.f;

    float controlledZoomFactor = fmin(radius, controlledZoomRadius) / controlledZoomRadius;

    radius += -offset * zoomSpeed * controlledZoomFactor;
    radius = glm::clamp(radius, minRadius, maxRadius);

    updatePosition();
}

void Camera::adjustPivot(const glm::vec2& offset)
{
    const glm::vec3 forward = glm::normalize(pivot - position);
    const glm::vec3 right   = glm::normalize(glm::cross(forward, Config::mesh.worldUp));
    const glm::vec3 up      = glm::normalize(glm::cross(right, forward));

    const float sensitivity = .002f * radius;

    pivot -= offset.x * sensitivity * right;
    pivot += offset.y * sensitivity * up;

    updatePosition();
}

void Camera::updatePosition()
{
    position.x = pivot.x + radius * sin(polar) * cos(azimuthal);
    position.y = pivot.y + radius * sin(polar) * sin(azimuthal);
    position.z = pivot.z + radius * cos(polar);
}
