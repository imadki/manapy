#pragma once

#include "./platform/window.hpp"
#include "./platform/inputManager.hpp"
#include "./resources/meshManager.hpp"
#include "./graphics/renderer.hpp"
#include "./scene/camera.hpp"
#include "./ui/editorUI.hpp"

class App {
  public:
    App();
    ~App();

    void run();

  private:
    Window       window;
    InputManager inputManager;
    MeshManager  meshManager;
    Renderer     renderer;
    Camera       camera;
    EditorUI     ui;
};
