#include "inputManager.hpp"

void InputManager::attach(Window& window)
{
    glfwWindow = window.getNative();

    window.setScrollExtCallback([this](double x, double y) { onScroll(x, y); });
}

InputState InputManager::consumeState()
{
    queryInput();

    InputState snapshot = state;
    state.reset();

    return snapshot;
}

void InputManager::queryInput()
{
    // NOTE: other input might be queried asynchronously

    // ─[ Mouse Buttons ]──────────────────────────────────────────────────
    state.mouseLeftDown  = (glfwGetMouseButton(glfwWindow, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    state.mouseRightDown = (glfwGetMouseButton(glfwWindow, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    state.mouseMiddleDown =
        (glfwGetMouseButton(glfwWindow, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);

    // ─[ Mouse Cursor ]───────────────────────────────────────────────────
    glm::dvec2 dCurrMousePos;
    glfwGetCursorPos(glfwWindow, &dCurrMousePos.x, &dCurrMousePos.y);

    glm::vec2 currMousePos = {(float)dCurrMousePos.x, (float)dCurrMousePos.y};
    glm::vec2 prevMousePos = state.mousePos;

    state.mouseDelta = {
        currMousePos.x - prevMousePos.x,
        currMousePos.y - prevMousePos.y,
    };

    state.mousePos = currMousePos;
}

void InputManager::onScroll(double x, double y) { state.scrollDelta += (float)y; }
