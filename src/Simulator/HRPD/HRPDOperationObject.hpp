#pragma once

#include "Simulator/PDTypeDef.hpp"

#include <igl/opengl/glfw/Viewer.h>
#include <spdlog/spdlog.h>

namespace PD {

enum OperationObjectType {
    NONE,
    GRIP_CUBE,
    COLLISION_CUBE,
    COLLISION_SPHERE,
};

const char* const OperationObjectTypeStrs[] = {
    "NONE",
    "GRIP_CUBE",
    "COLLISION_CUBE",
    "COLLISION_SPHERE"
};

// ========================== Base Object ==========================
struct OperationObject {
    PDPositions m_verts, m_initVerts;
    PDTriangles m_faces;
    PD3dVector m_center;
    PD3dVector m_length{ 10, 10, 10 };
    OperationObjectType m_type{ OperationObjectType::NONE };
    int m_curMove{ 0 };

    int m_meshID{ -1 };
    Eigen::Matrix4f m_gizmoT{ Eigen::Matrix4f::Identity() };

public:
    OperationObject() {}
    OperationObject(PD3dVector center, PD3dVector length)
        : m_center(center), m_length(length) {}
    // To init the verts position
    virtual void setVertsFromCenterAndLength() = 0;

public:
    void ApplyTransform(const Eigen::Matrix4d& T, Eigen::Matrix4f& gizmoT);
    void SetLength(PD3dVector length);
    void Reset();
    virtual void Step(double dt) = 0;
    virtual void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) = 0;
    void IGL_DelMesh(igl::opengl::glfw::Viewer* viewer) {
        if (m_meshID != -1) {
            spdlog::info(">>> OperationObject - IGL_DelMesh, meshID = {}", m_meshID);
            viewer->data(m_meshID).clear();
            viewer->erase_mesh(m_meshID);
        }
    }
};

struct GripObject : public OperationObject {
public:
    virtual bool ResolveGrip(PD3dVector& pos) = 0;
};

struct CollisionObject : public OperationObject {
public:
    virtual bool ResolveCollision(PD3dVector& pos) = 0;
};

// ========================== Specific Object ==========================

struct GripCube : public GripObject {
    OperationObjectType m_type{ OperationObjectType::GRIP_CUBE };

public:
    GripCube(PD3dVector center, PD3dVector length);
    GripCube(PDPositions verts);
    void setVertsFromCenterAndLength() override;

public:
    std::vector<unsigned int> getGripInds(PDPositions& V);
    void Step(double dt) override {}
    bool ResolveGrip(PD3dVector& pos) override;
    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) override;
};

struct CollisionCube : public CollisionObject {
    OperationObjectType m_type{ OperationObjectType::COLLISION_CUBE };
    // For translate
    PD3dVector m_translateDir;
    // For rotate
    std::pair<int, int> m_rotateEdgeInds;
    std::vector<int> m_rotatePlaneVertInd{ 4 };
    PD3dVector m_rotatePlaneNorm;
    bool m_isRotate{ false };
    bool m_isTranslate{ false };

    int m_maxMove = 1000;
    double m_moveVel{ 0 };

public:
    CollisionCube(PD3dVector center, PD3dVector length);
    CollisionCube(PDPositions verts);
    void setVertsFromCenterAndLength() override;
    void initTranslate(PD3dVector translateDir = { 0.0, 0.0, 0.0 }, double moveVel = 0.0);
    void initRotate(std::pair<int, int> rotateVertInd, double moveVel = 1.0);
    bool isInside(PD3dVector vert);

public:
    void Step(double dt) override;
    bool ResolveCollision(PD3dVector& pos) override;
    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) override;
};

struct CollisionSphere : public CollisionObject {
    OperationObjectType m_type{ OperationObjectType::COLLISION_SPHERE };
    PD3dVector m_translateDir;
    int m_maxMove = 1000;
    double m_moveVel{ 1.0 };

public:
    CollisionSphere(PD3dVector center, PD3dVector length, PD3dVector translateDir = { 0.0, 0.0, 0.0 }, double moveVel = 0.0);
    CollisionSphere(PDPositions verts, PDTriangles faces, PD3dVector translateDir = { 0.0, 0.0, 0.0 }, double moveVel = 0.0);
    void setHardEncodeSphere();
    void setVertsFromCenterAndLength() override;

public:
    void Step(double dt) override;
    bool ResolveCollision(PD3dVector& pos) override;
    void IGL_SetMesh(igl::opengl::glfw::Viewer* viewer) override;
};

}