#pragma once

#include <glm/glm.hpp>
#include "../platform/inputState.hpp"
#include "../ui/uiState.hpp"
#include "./cameraData.hpp"

class Camera {
  public:
    Camera();

    void update(const InputState& inputState, const UIState& uiState);

    const CameraData& getCameraData() const;

  private:
    CameraData data;

    glm::vec3 position;

    float radius;
    float polar;
    float azimuthal;

    float fov;
    float near;
    float far;

    glm::vec2 dragSensitivity;
    float     scrollSensitivity;

  private:
    void orbit(glm::vec2 offset);
    void adjustRadius(float offset);

    void updatePosition();
};
