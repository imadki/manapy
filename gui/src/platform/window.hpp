#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <functional>

class Window {
    using ResizeExtCallback = std::function<void(int, int)>;
    using ScrollExtCallback = std::function<void(double, double)>;

  public:
    void init();
    void shutdown();

    GLFWwindow* getNative();

    void setResizeExtCallback(ResizeExtCallback callback);
    void setScrollExtCallback(ScrollExtCallback callback);

    bool shouldClose() const;
    void pollEvents() const;

  private:
    GLFWwindow* glfwWindow = nullptr;

    // TODO: set to vector of callbacks to be called
    ResizeExtCallback resizeExtCallback = nullptr;
    ScrollExtCallback scrollExtCallback = nullptr;

  private:
    static void resizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
};
