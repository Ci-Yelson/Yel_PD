#pragma once

#include "PDTypeDef.hpp"
#include "Simulator/HRPD/HRPDOperationManager.hpp"
#include "igl/opengl/glfw/Viewer.h"

namespace PD {
struct PDSimulator {

public:
    // ----------------- UI-Operation -----------------
    OperationManager m_OpManager;

public:
    // todo
    virtual void LoadParamsAndApply() = 0;
    virtual void UpdateTimeStep() = 0;

    virtual void Step() = 0;
    virtual void Reset() = 0;

    // For update mesh data
    virtual void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) = 0;

    // For user interaction
    virtual const PDPositions& GetRestPositions() = 0;
    virtual const PDPositions& GetPositions() = 0;
    virtual const PDTriangles& GetTriangles() = 0;
};
}