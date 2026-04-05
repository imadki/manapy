#include "application.hpp"

Application::Application() : window(this->appName), renderer(this->appName, window) {}

void Application::run()
{
    while (!window.shouldClose()) {
        window.pollEvents();
        renderer.drawFrame();
    }
}
