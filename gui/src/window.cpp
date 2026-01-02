#include "window.hpp"

Window::Window(const char* title)
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    glfwWindow =
        glfwCreateWindow(Window::defaultWidth, Window::defaultHeight, title, nullptr, nullptr);

    glfwSetWindowSize(glfwWindow, Window::defaultWidth, Window::defaultHeight);

    glfwSetWindowSizeCallback(glfwWindow, Window::framebufferResizeCallback);
}

Window::~Window()
{
    glfwDestroyWindow(glfwWindow);
    glfwTerminate();
}

bool Window::shouldClose() { return glfwWindowShouldClose(glfwWindow); }
void Window::pollEvents() { glfwPollEvents(); }
void Window::waitEvents() { glfwWaitEvents(); }

void Window::setResizeUserPointer(bool* pResized)
{
    glfwSetWindowUserPointer(glfwWindow, pResized);
}

void Window::createSurface(VkInstance vkInstance, VkSurfaceKHR* pSurface)
{
    VK_CHECK(glfwCreateWindowSurface(vkInstance, glfwWindow, nullptr, pSurface));
}

void Window::getFramebufferSize(int* pWidth, int* pHeight)
{
    glfwGetFramebufferSize(glfwWindow, pWidth, pHeight);
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    if (glfwGetWindowUserPointer(window) == nullptr)
        throw std::runtime_error("Window resize user pointer not initialized!");

    bool* pResized = reinterpret_cast<bool*>(glfwGetWindowUserPointer(window));
    *pResized      = true;
}
