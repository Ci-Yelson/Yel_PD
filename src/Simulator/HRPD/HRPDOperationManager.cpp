#include "Simulator/HRPD/HRPDOperationManager.hpp"
#include "Simulator/HRPD/HRPDOperationObject.hpp"

namespace PD {

void OperationManager::AddOperationObject(std::string objectTypeStr, double length)
{
    auto& objectTypeStrs = OperationObjectTypeStrs;

    auto& V = m_mesh->m_positions;
    Eigen::RowVector3d mi = V.colwise().minCoeff();
    Eigen::RowVector3d mx = V.colwise().maxCoeff();
    if (length == -1) {
        length = std::min({ mx.x() - mi.x(), mx.y() - mi.y(), mx.z() - mi.z() });
    }

    spdlog::info(">>> OperationManager::AddOperationObject - length = {}", length);

    if (objectTypeStr == objectTypeStrs[OperationObjectType::NONE]) {
        return;
    }
    else if (objectTypeStr == objectTypeStrs[OperationObjectType::GRIP_CUBE]) {
        PD3dVector center = { 0.0, (mx.y() + mx.y() + length) * 0.5, 0.0 };
        PD3dVector length3d = { 2 * length, length, 2 * length };
        std::shared_ptr<GripCube> gripCubePtr = std::make_shared<GripCube>(center, length3d);
        m_operationObjects.push_back(gripCubePtr);
    }
    else if (objectTypeStr == objectTypeStrs[OperationObjectType::COLLISION_CUBE]) {
        PD3dVector center = { 0.0, (mx.y() + mx.y() + length) * 0.5, 0.0 };
        PD3dVector length3d = { 2 * length, length, 2 * length };
        std::shared_ptr<CollisionCube> collisionCubePtr = std::make_shared<CollisionCube>(center, length3d);
        m_operationObjects.push_back(collisionCubePtr);
    }
    else if (objectTypeStr == objectTypeStrs[OperationObjectType::COLLISION_SPHERE]) {

        PD3dVector center = { 0, (mx.y() + mx.y() + length) * 0.5, 0 };
        PD3dVector length3d = { length, length, length };
        std::shared_ptr<CollisionSphere> collisionSpherePtr = std::make_shared<CollisionSphere>(center, length3d);
        m_operationObjects.push_back(collisionSpherePtr);
    }
    else {
        spdlog::warn("Not support entity type!");
    }

    m_selectObjectIndex = m_operationObjects.size() - 1;
}

void OperationManager::DelOperationObject(igl::opengl::glfw::Viewer* viewer)
{
    if (m_selectObjectIndex == -1) {
        spdlog::info("No selecting object!");
        return;
    }
    m_operationObjects[m_selectObjectIndex]->IGL_DelMesh(viewer);
    m_operationObjects.erase(m_operationObjects.begin() + m_selectObjectIndex);
    m_selectObjectIndex = -1;
}

void OperationManager::Step(double dt)
{
    for (auto& obj : m_operationObjects) obj->Step(dt);
    // spdlog::info(">>> OperationManager::Step() - After");
}

void OperationManager::IGL_SetMesh(igl::opengl::glfw::Viewer* viewer)
{
    for (auto& obj : m_operationObjects) obj->IGL_SetMesh(viewer);
    // spdlog::info(">>> OperationManager::IGL_SetMesh() - After");
}

// ================================= Presets =================================
void OperationManager::OperationSetupSphere()
{
    spdlog::info("### OperationSetupSphere");

    auto& V = m_mesh->m_positions;
    Eigen::RowVector3d originMi = V.colwise().minCoeff();
    Eigen::RowVector3d originMx = V.colwise().maxCoeff();

    double xlen = (originMx.x() - originMi.x()) * 0.3;

    {
        // grip cube [floor]
        double thickness = 3.0f;
        PD3dVector center = { (originMi.x() + originMx.x()) * 0.5, originMi.y() + 0.5 * thickness, (originMi.z() + originMx.z()) * 0.5 };
        PD3dVector length3d = { originMx.x() - originMi.x(), thickness, originMx.z() - originMi.z() };
        std::shared_ptr<GripCube> gripCubePtr = std::make_shared<GripCube>(center, length3d);
        m_operationObjects.push_back(gripCubePtr);
    }

    {
        // left
        double offsetX = -50;
        double offsetY = std::max(0.0, originMx.y() + xlen) - 20;
        double offsetZ = -10;
        PD3dVector center = { offsetX, offsetY, offsetZ };
        PD3dVector length3d = { xlen, xlen, xlen };
        PD3dVector dir = { 0, -1, 0 };
        std::shared_ptr<CollisionObject> colObjPtr = std::make_shared<CollisionSphere>(center, length3d, dir, 0.01f);
        m_operationObjects.push_back(colObjPtr);
    }

    {
        // right
        double offsetX = 100;
        double offsetY = std::max(0.0, originMx.y() + xlen) - 20;
        double offsetZ = -10;
        PD3dVector center = { offsetX, offsetY, offsetZ };
        PD3dVector length3d = { xlen, xlen, xlen };
        PD3dVector dir = { 0, -1, 0 };
        std::shared_ptr<CollisionObject> colObjPtr = std::make_shared<CollisionSphere>(center, length3d, dir, 0.01f);
        m_operationObjects.push_back(colObjPtr);
    }
}

void OperationManager::OperationSetupCube()
{
    spdlog::info("### OperationSetupCube");

    auto& V = m_mesh->m_positions;
    Eigen::RowVector3d mi = V.colwise().minCoeff();
    Eigen::RowVector3d mx = V.colwise().maxCoeff();

    double thickness = 5.0;

    {
        // up
        PD3dVector center = { (mx.x() + mi.x()) * 0.5, mx.y() + 0.5 * thickness, (mx.z() + mi.z()) * 0.5 };
        PD3dVector length3d = { mx.x() - mi.x(), thickness, mx.z() - mi.z() };
        std::shared_ptr<CollisionCube> colObjPtr = std::make_shared<CollisionCube>(center, length3d);
        PD3dVector dir = { 0.0, -1.0, 0.0 };
        colObjPtr->initTranslate(dir, 0.01);
        m_operationObjects.push_back(colObjPtr);
    }
}

void OperationManager::OperationSetupRotation()
{
    spdlog::info("### OperationSetupRotation");
    auto& V = m_mesh->m_positions;
    Eigen::RowVector3d originMi = V.colwise().minCoeff();
    Eigen::RowVector3d originMx = V.colwise().maxCoeff();

    double left = -10.0, right = 10.0, thickness = 5.0;
    double gripThickness = (originMx.y() - originMi.y()) * 0.5;

    {
        // grip cube
        auto mi = originMi, mx = originMx;

        PD3dVector center = { (left + right) * 0.5, mx.y() + 0.5 * gripThickness, (mx.z() + mi.z()) * 0.5 };
        PD3dVector length3d = { right - left, gripThickness, mx.z() - mi.z() };
        std::shared_ptr<GripCube> gripCubePtr = std::make_shared<GripCube>(center, length3d);
        m_operationObjects.push_back(gripCubePtr);
    }

    {
        // left
        auto mi = originMi, mx = originMx;
        mx.x() = std::min(left, mx.x());

        PD3dVector center = { (mx.x() + mi.x()) * 0.5, mi.y() - 0.5 * thickness, (mx.z() + mi.z()) * 0.5 };
        PD3dVector length3d = { mx.x() - mi.x(), thickness, mx.z() - mi.z() };

        std::shared_ptr<CollisionCube> colObjPtr = std::make_shared<CollisionCube>(center, length3d);
        std::pair<int, int> rotateVertInd = { 2, 1 };
        colObjPtr->initRotate(rotateVertInd, 0.01);
        m_operationObjects.push_back(colObjPtr);
    }

    {
        // right
        auto mi = originMi, mx = originMx;
        mi.x() = std::max(right, mi.x());

        PD3dVector center = { (mx.x() + mi.x()) * 0.5, mi.y() - 0.5 * thickness, (mx.z() + mi.z()) * 0.5 };
        PD3dVector length3d = { mx.x() - mi.x(), thickness, mx.z() - mi.z() };

        std::shared_ptr<CollisionCube> colObjPtr = std::make_shared<CollisionCube>(center, length3d);
        std::pair<int, int> rotateVertInd = { 0, 3 };
        colObjPtr->initRotate(rotateVertInd, 0.01);
        m_operationObjects.push_back(colObjPtr);
    }
}

}