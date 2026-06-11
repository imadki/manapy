#pragma once

#include <glm/glm.hpp>

struct InputState {
    glm::vec2 mousePos;
    glm::vec2 mouseDelta;
    float     scrollDelta;

    bool mouseLeftDown;
    bool mouseRightDown;
    bool mouseMiddleDown;

    InputState();
    void reset();
};
