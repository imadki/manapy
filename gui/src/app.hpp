#pragma once

#include "renderer.hpp"
#include "window.hpp"

class App {
  public:
    App();
    ~App();

    void run();

  private:
    const char* appName = "ManapyGUI";

    // CRITICAL initialization order: window -> renderer
    Window   window;
    Renderer renderer;
};
