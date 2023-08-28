#pragma once

#include "PDTypeDef.hpp"
#include "igl/opengl/glfw/Viewer.h"

namespace PD {
struct PDSimulator {
    virtual void LoadParamsAndApply() = 0;
    virtual void UpdateTimeStep() = 0;

    virtual void Step() = 0;
    virtual void Reset() = 0;

    virtual void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) = 0;

    virtual const PDPositions& GetRestPositions() = 0;
    virtual const PDPositions& GetPositions() = 0;
    virtual const PDTriangles& GetTriangles() = 0;
};
}