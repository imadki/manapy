#include "window.hpp"
#include "../common/config.hpp"
#include <utility>

void Window::init()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    glfwWindow = glfwCreateWindow(Config::window.width,
                                  Config::window.height,
                                  Config::window.title,
                                  nullptr,
                                  nullptr);

    glfwSetWindowUserPointer(glfwWindow, this);

    glfwSetWindowSizeCallback(glfwWindow, Window::resizeCallback);
    glfwSetScrollCallback(glfwWindow, Window::scrollCallback);
}

void Window::shutdown()
{
    glfwDestroyWindow(glfwWindow);
    glfwTerminate();
}

GLFWwindow* Window::getNative() { return glfwWindow; }

void Window::setResizeExtCallback(ResizeExtCallback callback)
{
    resizeExtCallback = std::move(callback);
}

void Window::setScrollExtCallback(ScrollExtCallback callback)
{
    scrollExtCallback = std::move(callback);
}

bool Window::shouldClose() const { return glfwWindowShouldClose(glfwWindow); }
void Window::pollEvents() const { glfwPollEvents(); }

void Window::resizeCallback(GLFWwindow* window, int width, int height)
{
    Window* instance = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

    if (instance->resizeExtCallback) {
        instance->resizeExtCallback(width, height);
    }
}

void Window::scrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
    Window* instance = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

    if (instance->scrollExtCallback) {
        instance->scrollExtCallback(xOffset, yOffset);
    }
}
