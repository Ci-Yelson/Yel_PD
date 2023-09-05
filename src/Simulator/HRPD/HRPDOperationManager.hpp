#pragma once

#include "Simulator/HRPD/HRPDOperationObject.hpp"
#include "Simulator/HRPD/HRPDTetMesh.hpp"

namespace PD {

const char* const OperationPresets[] = {
    "NONE",
    "PRESET_SPHERE",
    "PRESET_CUBE",
    "PRESET_ROATION"
};

struct OperationManager {
    std::vector<std::shared_ptr<OperationObject>> m_operationObjects;
    std::shared_ptr<HRPDTetMesh> m_mesh;

    int m_selectObjectIndex = -1;
    int m_presetIndex = 0;

public:
    void Setup(std::shared_ptr<HRPDTetMesh> tetmesh)
    {
        m_mesh = tetmesh;
    }
    void Reset()
    {
        m_selectObjectIndex = -1;
        for (auto& obj : m_operationObjects) obj->Reset();
    }
    void Clear(igl::opengl::glfw::Viewer* viewer)
    {
        spdlog::info(">>> OperationManager - Clear - Before, obj size = {}", m_operationObjects.size());
        m_selectObjectIndex = -1;
        while (m_operationObjects.size()) {
            auto& obj = m_operationObjects.back();
            obj->IGL_DelMesh(viewer);
            m_operationObjects.pop_back();
        }
        spdlog::info(">>> OperationManager - Clear - After");
    }
    void SetupPreset(igl::opengl::glfw::Viewer* viewer)
    {
        Clear(viewer);
        if (m_presetIndex == 0)
            ;
        else if (m_presetIndex == 1)
            OperationSetupSphere();
        else if (m_presetIndex == 2)
            OperationSetupCube();
        else if (m_presetIndex == 3)
            OperationSetupRotation();
    }

public:
    void AddOperationObject(std::string objectTypeStr, double length = -1);
    void DelOperationObject(igl::opengl::glfw::Viewer* viewer);
    void Step(double dt);
    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer);

public:
    // presetIndex = 0 -> NONE
    // presetIndex = 1
    void OperationSetupSphere();
    // presetIndex = 2
    void OperationSetupCube();
    // presetIndex = 3
    void OperationSetupRotation();
};

}