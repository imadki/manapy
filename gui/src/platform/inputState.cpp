#include "inputState.hpp"

InputState::InputState() : mousePos(0.f, 0.f) { reset(); }

void InputState::reset()
{
    mouseDelta  = {0.f, 0.f};
    scrollDelta = 0.0f;

    mouseLeftDown   = false;
    mouseRightDown  = false;
    mouseMiddleDown = false;
}
