#include "camera.hpp"

Camera::Camera()
    : position(::modelPivot + glm::vec3(1.f, 3.f, 2.f)), fov(60.f), near(.01f), far(100.f),
      dragSensitivity(.8f, .6f), scrollSensitivity(.5f)
{
    glm::vec3 offset = position - ::modelPivot;

    radius    = glm::length(offset);
    azimuthal = atan2(position.y, position.x);
    polar     = acos(position.z / radius);
}

void Camera::orbit(glm::vec2 dragOffset)
{
    const static float minPolar = glm::radians(10.f);
    const static float maxPolar = glm::radians(170.f);

    if (!glm::length(dragOffset)) return;

    azimuthal += glm::radians(-dragOffset.x * dragSensitivity.x);

    polar += glm::radians(-dragOffset.y * dragSensitivity.y);
    polar = glm::clamp(polar, minPolar, maxPolar);

    updatePosition();
};

void Camera::adjustRadius(float scrollOffset)
{
    const static float minRadius = 1.f;
    const static float maxRadius = far;

    const static float controlledZoomRadius = 10.f;

    float controlledZoomFactor = fmin(radius, controlledZoomRadius) / controlledZoomRadius;

    radius += -scrollOffset * scrollSensitivity * controlledZoomFactor;
    radius = glm::clamp(radius, minRadius, maxRadius);

    updatePosition();
}

glm::mat4 Camera::getViewMat() const { return glm::lookAt(position, ::modelPivot, ::worldUp); }

glm::mat4 Camera::getProjMat(float aspectRatio) const
{
    return glm::perspective(glm::radians(fov), aspectRatio, near, far);
}

void Camera::updatePosition()
{
    position.x = ::modelPivot.x + radius * sin(polar) * cos(azimuthal);
    position.y = ::modelPivot.y + radius * sin(polar) * sin(azimuthal);
    position.z = ::modelPivot.z + radius * cos(polar);
}
