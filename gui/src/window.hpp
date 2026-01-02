#pragma once

#include "common.hpp"

class Window {
  public:
    Window(const char* title);
    ~Window();

    bool shouldClose();
    void pollEvents();
    void waitEvents();

    void setResizeUserPointer(bool* pResized);

    void createSurface(VkInstance vkInstance, VkSurfaceKHR* pSurface);
    void getFramebufferSize(int* pWidth, int* pHeight);

  private:
    static const int defaultWidth  = 800;
    static const int defaultHeight = 600;

    GLFWwindow* glfwWindow;

  private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};
