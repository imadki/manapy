#include "./editorUI.hpp"
#include "imgui.h"
#include <map>
#include <iostream>
#include <magic_enum.hpp>
#include <string>
#include <sys/stat.h>

void EditorUI::init(const VulkanContext&          vkCtx,
                    GLFWwindow*                   glfwWindow,
                    ImGui_ImplVulkan_PipelineInfo imGuiPipelineInfo)
{
    backend.init(vkCtx, glfwWindow, imGuiPipelineInfo);

    meshFileDialog.SetTitle("Load mesh");
    meshFileDialog.SetTypeFilters({".msh"});

    style();
}

void EditorUI::shutdown() { backend.shutdown(); }

void EditorUI::build()
{
    backend.newFrame();
    buildDockLayout();

    for (auto const& panel : magic_enum::enum_values<Panel>())
        buildDockPanel(panel);

    // TODO: Set per panel window flags to set on buildDockPanel
    // e.g:ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
}

void EditorUI::insertMeshViewTexture(VkDescriptorSet meshViewTexture)
{
    if (ImGui::Begin(getLabel(Panel::Viewport).data()))
        ImGui::Image((ImTextureID)meshViewTexture, ImGui::GetContentRegionAvail());
    ImGui::End();
}

const UIState& EditorUI::getState() const { return state; }

void EditorUI::buildDockLayout()
{
    const char*    dockspaceStrId = "Dockspace";
    ImGuiID        dockspaceId    = ImGui::GetID(dockspaceStrId);
    ImGuiViewport* viewport       = ImGui::GetMainViewport();

    // Create settings
    if (ImGui::DockBuilderGetNode(dockspaceId) == nullptr) {
        ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->Size);

        const float leftRatio   = .2f;
        const float rightRatio  = .2f;
        const float bottomRatio = .2f;

        ImGuiID dockIdLeft, dockIdMain;
        ImGui::DockBuilderSplitNode(dockspaceId,
                                    ImGuiDir_Left,
                                    leftRatio,
                                    &dockIdLeft,
                                    &dockIdMain);

        ImGuiID dockIdRight;
        ImGui::DockBuilderSplitNode(dockIdMain,
                                    ImGuiDir_Right,
                                    rightRatio / (1 - leftRatio),
                                    &dockIdRight,
                                    &dockIdMain);

        ImGuiID dockIdBottom;
        ImGui::DockBuilderSplitNode(dockIdMain,
                                    ImGuiDir_Down,
                                    bottomRatio,
                                    &dockIdBottom,
                                    &dockIdMain);

        ImGui::DockBuilderDockWindow(getLabel(Panel::Simulation).data(), dockIdLeft);
        ImGui::DockBuilderDockWindow(getLabel(Panel::Domain).data(), dockIdLeft);
        ImGui::DockBuilderDockWindow(getLabel(Panel::Viewport).data(), dockIdMain);
        ImGui::DockBuilderDockWindow(getLabel(Panel::ViewportSettings).data(), dockIdRight);
        ImGui::DockBuilderDockWindow(getLabel(Panel::Console).data(), dockIdBottom);

        ImGui::DockBuilderFinish(dockspaceId);
    }

    // Submit dockspace
    ImGui::DockSpaceOverViewport(dockspaceId, viewport, ImGuiDockNodeFlags_PassthruCentralNode);
}

void EditorUI::buildDockPanel(Panel panel)
{
    if (ImGui::Begin(getLabel(panel).data())) {
        switch (panel) {
            // ╭─────────────────────────────────────────────────────────╮
            // │                    Simulation Panel                     │
            // ╰─────────────────────────────────────────────────────────╯
        case Panel::Simulation: {
        } break;

            // ╭─────────────────────────────────────────────────────────╮
            // │                      Domain Panel                       │
            // ╰─────────────────────────────────────────────────────────╯
        case Panel::Domain: {
            ImGui::SeparatorText("Mesh");

            // ─[ Mesh Filename ]──────────────────────────────────────────────────
            bool        hasMesh = !state.meshFile.path.empty();
            std::string meshFilename =
                hasMesh ? state.meshFile.path.filename().string() : "No mesh loaded...";

            ImGui::BeginDisabled();
            ImGui::InputText("##meshFilename",
                             meshFilename.data(),
                             meshFilename.size(),
                             ImGuiInputTextFlags_ReadOnly);

            if (hasMesh && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::BeginTooltip();
                ImGui::TextDisabled("%s", state.meshFile.path.c_str());
                ImGui::EndTooltip();
            }

            ImGui::EndDisabled();

            ImGui::SameLine();
            // ─[ Mesh Load ]──────────────────────────────────────────────────────
            if (ImGui::Button("Browse")) meshFileDialog.Open();
            meshFileDialog.Display();

            if (meshFileDialog.HasSelected()) {
                state.meshFile.isSelected = true;
                state.meshFile.path       = meshFileDialog.GetSelected();
                meshFileDialog.ClearSelected();
            }
            else {
                state.meshFile.isSelected = false;
            }
        } break;

            // ╭─────────────────────────────────────────────────────────╮
            // │                     Viewport Panel                      │
            // ╰─────────────────────────────────────────────────────────╯
        case Panel::Viewport: {
            // ─[ State Update ]───────────────────────────────────────────────────
            state.meshView.isFocused = ImGui::IsWindowFocused();
            state.meshView.isHovered = ImGui::IsWindowHovered();

            ImVec2   unscaledSize = ImGui::GetContentRegionAvail();
            ImGuiIO& io           = ImGui::GetIO();

            state.meshView.size.x = unscaledSize.x * io.DisplayFramebufferScale.x;
            state.meshView.size.y = unscaledSize.y * io.DisplayFramebufferScale.y;
        } break;

            // ╭─────────────────────────────────────────────────────────╮
            // │                 Viewport Settings Panel                 │
            // ╰─────────────────────────────────────────────────────────╯
        case Panel::ViewportSettings: {
            if (ImGui::CollapsingHeader("Camera")) {
                ImGui::SeparatorText("View");
                ImGui::DragFloat("FOV",
                                 &state.meshView.cameraFov,
                                 0.5f,
                                 20.f,
                                 90.f,
                                 "%.0f",
                                 ImGuiSliderFlags_AlwaysClamp);

                ImGui::SeparatorText("Sensitivity");
                ImGui::DragFloat2("Orbit Speed",
                                  (float*)&state.meshView.orbitSpeed,
                                  .01f,
                                  .1f,
                                  2.f,
                                  "%.2f",
                                  ImGuiSliderFlags_AlwaysClamp);
                helpMarker("(X, Y)");

                ImGui::DragFloat("Zoom Speed",
                                 &state.meshView.zoomSpeed,
                                 .01f,
                                 .1f,
                                 2.f,
                                 "%.2f",
                                 ImGuiSliderFlags_AlwaysClamp);

                ImGui::Spacing();
            }

            if (ImGui::CollapsingHeader("Visualization")) {
                ImGui::SeparatorText("Render Style");
                // ─[ Polygon Mode ]───────────────────────────────────────────────────
                static const std::map<VkPolygonMode, const char*> modeNames = {
                    {VK_POLYGON_MODE_FILL, "Fill"},
                    {VK_POLYGON_MODE_LINE, "Line"},
                    {VK_POLYGON_MODE_POINT, "Point"},
                };

                const char* preview = modeNames.at(state.meshView.polygoneMode);

                if (ImGui::BeginCombo("Polygon Mode", preview)) {
                    for (const auto& [mode, name] : modeNames) {
                        bool isSelected = (state.meshView.polygoneMode == mode);

                        if (ImGui::Selectable(name, isSelected)) state.meshView.polygoneMode = mode;

                        if (isSelected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }
        } break;

            // ╭─────────────────────────────────────────────────────────╮
            // │                      Console Panel                      │
            // ╰─────────────────────────────────────────────────────────╯
        case Panel::Console: {
        } break;
        }
    }
    ImGui::End();
}

void EditorUI::style()
{
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGuiStyle& style = ImGui::GetStyle();

    ImFont* font = io.Fonts->AddFontFromFileTTF(ASSETS_DIR "/fonts/Rubik-Regular.ttf", 16.f);
    if (!font) std::cerr << "Failed to load editor UI font" << std::endl;

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

constexpr std::string_view EditorUI::getLabel(Panel panel)
{
    switch (panel) {
    case Panel::Simulation: return "Simulation";
    case Panel::Domain: return "Domain";
    case Panel::Viewport: return "Viewport";
    case Panel::ViewportSettings: return "Viewport Settings";
    case Panel::Console: return "Console";
    default: return "Unknown Panel";
    }
}
