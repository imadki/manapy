#include "window.hpp"

Window::Window(const char* title)
    : pFrameBufferResized(std::make_shared<bool>(false)), scrollOffset(0.f)
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    glfwWindow =
        glfwCreateWindow(Window::defaultWidth, Window::defaultHeight, title, nullptr, nullptr);
    glfwSetWindowSize(glfwWindow, Window::defaultWidth, Window::defaultHeight);

    glfwSetWindowUserPointer(glfwWindow, this);

    glfwSetWindowSizeCallback(glfwWindow, Window::framebufferResizeCallback);
    glfwSetScrollCallback(glfwWindow, Window::scrollCallback);
}

Window::~Window()
{
    glfwDestroyWindow(glfwWindow);
    glfwTerminate();
}

bool Window::shouldClose() { return glfwWindowShouldClose(glfwWindow); }
void Window::pollEvents() { glfwPollEvents(); }
void Window::waitEvents() { glfwWaitEvents(); }

void Window::createSurface(VkInstance vkInstance, VkSurfaceKHR* pSurface)
{
    VK_CHECK(glfwCreateWindowSurface(vkInstance, glfwWindow, nullptr, pSurface));
}

void Window::getFramebufferSize(int* pWidth, int* pHeight)
{
    glfwGetFramebufferSize(glfwWindow, pWidth, pHeight);
}

glm::vec2 Window::getDragOffset()
{
    static glm::dvec2 lastMousePos;

    glm::dvec2 currMousePos;
    glfwGetCursorPos(glfwWindow, &currMousePos.x, &currMousePos.y);

    if (glfwGetMouseButton(glfwWindow, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
        lastMousePos = currMousePos;
        return glm::vec2{0.f};
    }

    glm::vec2 dragOffset{
        static_cast<float>(currMousePos.x - lastMousePos.x),
        static_cast<float>(currMousePos.y - lastMousePos.y),
    };

    lastMousePos = currMousePos;
    return dragOffset;
}

void  Window::addScrollOffset(float offset) { scrollOffset += offset; }
float Window::getAndResetScrollOffset()
{
    float offset = scrollOffset;
    scrollOffset = 0.f;

    return offset;
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    Window* instance = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

    *instance->pFrameBufferResized = true;
}

void Window::scrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
    Window* instance = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

    instance->addScrollOffset(yOffset);
}
