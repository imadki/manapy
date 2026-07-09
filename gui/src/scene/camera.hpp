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
    glm::vec3 pivot;

    float radius;
    float polar;
    float azimuthal;

    float fov  = 60.f;
    float near = .01f;
    float far  = 100.f;

    glm::vec2 orbitSpeed = glm::vec2(.8f, .6f);
    float     zoomSpeed  = .5f;

  private:
    void orbit(const glm::vec2& offset);
    void adjustRadius(float offset);
    void adjustPivot(const glm::vec2& offset);

    void updatePosition();
};
