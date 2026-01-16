#pragma once

#include "common.hpp"

class Window {
  public:
    std::shared_ptr<bool> pFrameBufferResized;

  public:
    Window(const char* title);
    ~Window();

    bool shouldClose();
    void pollEvents();
    void waitEvents();

    void createSurface(VkInstance vkInstance, VkSurfaceKHR* pSurface);
    void getFramebufferSize(int* pWidth, int* pHeight);

    glm::vec2 getDragOffset();

    void  addScrollOffset(float offset);
    float getAndResetScrollOffset();

  private:
    static const int defaultWidth  = 800;
    static const int defaultHeight = 600;

    GLFWwindow* glfwWindow;
    float       scrollOffset;

  private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
};
