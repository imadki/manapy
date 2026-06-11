#pragma once

#include "./inputState.hpp"
#include "../platform/window.hpp"

class InputManager {
  public:
    void attach(Window& window);

    InputState consumeState();

  private:
    GLFWwindow* glfwWindow;
    InputState  state;

  private:
    void queryInput();
    void onScroll(double x, double y);
};
