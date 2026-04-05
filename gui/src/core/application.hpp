#pragma once

#include "../graphics/renderer.hpp"
#include "window.hpp"

class Application {
  public:
    Application();

    void run();

  private:
    const char* appName = "ManapyGUI";

    // CRITICAL initialization order: window -> renderer
    Window   window;
    Renderer renderer;
};
