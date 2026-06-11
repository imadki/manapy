#include "app.hpp"

App::App()
{
    window.init();

    inputManager.attach(window);

    renderer.init(window.getNative());
    renderer.attach(window);

    meshManager.init(renderer.getVulkanContext());

    ui.init(renderer.getVulkanContext(), window.getNative(), renderer.getImGuiPipelineInfo());
    renderer.initMeshViewTextureDesc();
}

App::~App()
{
    renderer.deviceWaitIdle();

    renderer.clearMeshViewTextureDesc();
    ui.shutdown();
    meshManager.shutdown(renderer.getVulkanContext());
    renderer.shutdown();
    window.shutdown();
}

void App::run()
{
    while (!window.shouldClose()) {
        window.pollEvents();

        if (renderer.beginFrame()) {
            ui.build();

            const auto uiState    = ui.getState();
            const auto inputState = inputManager.consumeState();

            meshManager.update(renderer.getVulkanContext(), uiState);
            camera.update(inputState, uiState);
            renderer.update(uiState);

            ui.insertMeshViewTexture(renderer.getMeshViewTextureDesc());

            renderer.drawFrame(uiState, camera.getCameraData(), meshManager.getMeshData());
        }
    }
}
