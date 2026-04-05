#include "editorUI.hpp"

void EditorUI::init(ImGui_ImplVulkan_InitInfo* initInfo, GLFWwindow* glfwWindow)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);
    ImGui_ImplVulkan_Init(initInfo);

    meshFileDialog.SetTitle("Load mesh");
    meshFileDialog.SetTypeFilters({".msh"});

    style();
}

void EditorUI::cleanup()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void EditorUI::draw(VkCommandBuffer commandBuffer, VkDescriptorSet meshViewTexture)
{
    build(meshViewTexture);

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

VkExtent2D EditorUI::getMeshViewportExtent() const
{
    uint32_t width  = (meshViewportSize.x > 0) ? static_cast<uint32_t>(meshViewportSize.x) : 0;
    uint32_t height = (meshViewportSize.y > 0) ? static_cast<uint32_t>(meshViewportSize.y) : 0;

    return VkExtent2D{width, height};
}

bool EditorUI::isMeshViewportFocused() const { return meshViewportFocused; }
bool EditorUI::isMeshViewportHovered() const { return meshViewportHovered; }

bool EditorUI::hasSelectedMesh(std::string* filePath) const
{
    *filePath = meshFileDialog.GetSelected().string();
    return meshFileDialog.HasSelected();
}
void EditorUI::clearMeshSelection() { meshFileDialog.ClearSelected(); }

void EditorUI::build(VkDescriptorSet meshViewTexture)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiID dockspaceId = ImGui::DockSpaceOverViewport();

    static bool firstTime = true;
    if (firstTime) {
        firstTime = false;

        ImGui::DockBuilderRemoveNode(dockspaceId);
        ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->WorkSize);

        ImGuiID dock_left, dock_remaining, dock_right, dock_center, dock_right_top,
            dock_right_bottom;

        ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Left, 0.20f, &dock_left, &dock_remaining);
        ImGui::DockBuilderSplitNode(dock_remaining,
                                    ImGuiDir_Right,
                                    0.25f,
                                    &dock_right,
                                    &dock_center);
        ImGui::DockBuilderSplitNode(dock_right,
                                    ImGuiDir_Up,
                                    0.50f,
                                    &dock_right_top,
                                    &dock_right_bottom);

        ImGui::DockBuilderDockWindow("Settings", dock_left);
        ImGui::DockBuilderDockWindow("Viewport", dock_center);
        ImGui::DockBuilderDockWindow("Properties", dock_right_top);
        ImGui::DockBuilderDockWindow("Console", dock_right_bottom);

        ImGui::DockBuilderFinish(dockspaceId);
    }

    ImGui::Begin("Settings");
    ImGui::Text("Settings panel");
    if (ImGui::Button("Load mesh")) meshFileDialog.Open();
    meshFileDialog.Display();
    ImGui::End();

    ImGui::Begin("Properties");
    ImGui::Text("Properties panel");
    ImGui::End();

    ImGui::Begin("Console");
    ImGui::Text("Console panel");
    ImGui::End();

    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    meshViewportFocused = ImGui::IsWindowFocused();
    meshViewportHovered = ImGui::IsWindowHovered();
    meshViewportSize    = ImGui::GetContentRegionAvail();
    ImGui::Image((ImTextureID)meshViewTexture, meshViewportSize);
    ImGui::End();
}

void EditorUI::style()
{
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGuiStyle& style = ImGui::GetStyle();

    // - Spacing/Padding
    style.WindowPadding    = ImVec2(12.f, 12.f);
    style.FramePadding     = ImVec2(12.f, 4.f);
    style.ItemSpacing      = ImVec2(8.f, 8.f);
    style.ItemInnerSpacing = ImVec2(4.f, 4.f);
    style.IndentSpacing    = 20.f;
    style.ScrollbarSize    = 10.f;
    style.GrabMinSize      = 10.f;

    // - Rounding
    style.WindowRounding    = 6.f;
    style.ChildRounding     = 6.f;
    style.FrameRounding     = 4.f;
    style.GrabRounding      = 4.f;
    style.PopupRounding     = 6.f;
    style.ScrollbarRounding = 6.f;
    style.TabRounding       = 4.0f;

    // - Trees
    style.TreeLinesFlags    = ImGuiTreeNodeFlags_DrawLinesFull;
    style.TreeLinesSize     = 1.f;
    style.TreeLinesRounding = 3.f;

    // - Border
    style.FrameBorderSize = 1.f;
    style.PopupBorderSize = 1.f;

    // - Hover
    style.HoverFlagsForTooltipMouse = ImGuiHoveredFlags_DelayNone | ImGuiHoveredFlags_Stationary;
    style.HoverFlagsForTooltipNav   = ImGuiHoveredFlags_DelayNone | ImGuiHoveredFlags_NoSharedDelay;

    // - Colors
    ImVec4* colors = ImGui::GetStyle().Colors;

    colors[ImGuiCol_Text]                      = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]              = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]                  = ImVec4(0.14f, 0.14f, 0.14f, 0.94f);
    colors[ImGuiCol_ChildBg]                   = ImVec4(0.00f, 0.00f, 0.00f, 0.31f);
    colors[ImGuiCol_PopupBg]                   = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_Border]                    = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_BorderShadow]              = ImVec4(0.20f, 0.20f, 0.20f, 0.39f);
    colors[ImGuiCol_FrameBg]                   = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]            = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_FrameBgActive]             = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_TitleBg]                   = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_TitleBgActive]             = ImVec4(0.27f, 0.27f, 0.27f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]          = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg]                 = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]               = ImVec4(0.35f, 0.35f, 0.35f, 0.59f);
    colors[ImGuiCol_ScrollbarGrab]             = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]      = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]       = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_CheckMark]                 = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrab]                = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]          = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
    colors[ImGuiCol_Button]                    = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_ButtonHovered]             = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_ButtonActive]              = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
    colors[ImGuiCol_Header]                    = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_HeaderHovered]             = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_HeaderActive]              = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
    colors[ImGuiCol_Separator]                 = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_SeparatorHovered]          = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_SeparatorActive]           = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
    colors[ImGuiCol_ResizeGrip]                = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_ResizeGripHovered]         = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_ResizeGripActive]          = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
    colors[ImGuiCol_InputTextCursor]           = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TabHovered]                = ImVec4(0.27f, 0.27f, 0.27f, 1.00f);
    colors[ImGuiCol_Tab]                       = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_TabSelected]               = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
    colors[ImGuiCol_TabSelectedOverline]       = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
    colors[ImGuiCol_TabDimmed]                 = ImVec4(0.20f, 0.20f, 0.20f, 0.78f);
    colors[ImGuiCol_TabDimmedSelected]         = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PlotLines]                 = ImVec4(0.63f, 0.63f, 0.63f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]          = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram]             = ImVec4(0.63f, 0.63f, 0.63f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]      = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]             = ImVec4(0.27f, 0.27f, 0.27f, 1.00f);
    colors[ImGuiCol_TableBorderStrong]         = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_TableBorderLight]          = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_TableRowBg]                = ImVec4(0.04f, 0.04f, 0.04f, 0.39f);
    colors[ImGuiCol_TableRowBgAlt]             = ImVec4(0.24f, 0.24f, 0.24f, 0.39f);
    colors[ImGuiCol_TextLink]                  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]            = ImVec4(0.39f, 0.39f, 0.39f, 0.39f);
    colors[ImGuiCol_TreeLines]                 = ImVec4(0.39f, 0.39f, 0.39f, 0.78f);
    colors[ImGuiCol_DragDropTarget]            = ImVec4(0.39f, 0.39f, 0.39f, 0.78f);
    colors[ImGuiCol_NavCursor]                 = ImVec4(0.39f, 0.39f, 0.39f, 0.78f);
    colors[ImGuiCol_NavWindowingHighlight]     = ImVec4(0.39f, 0.39f, 0.39f, 0.78f);
    colors[ImGuiCol_NavWindowingDimBg]         = ImVec4(0.35f, 0.35f, 0.35f, 0.27f);
    colors[ImGuiCol_ModalWindowDimBg]          = ImVec4(0.35f, 0.35f, 0.35f, 0.27f);
}

void EditorUI::helpMarker(const char* desc)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}
