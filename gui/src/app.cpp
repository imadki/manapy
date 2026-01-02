#include "app.hpp"

App::App() : window(this->appName), renderer(this->appName, window) {}

App::~App() {}

void App::run()
{
    while (!window.shouldClose()) {
        window.pollEvents();
        renderer.drawFrame();
    }
}
