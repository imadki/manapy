#pragma once

#include "common.hpp"

class Camera {
  public:
    Camera();

    void orbit(glm::vec2 dragOffset);
    void adjustRadius(float scrollOffset);

    glm::mat4 getViewMat() const;
    glm::mat4 getProjMat(float aspectRatio) const;

  private:
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
    void updatePosition();
};
