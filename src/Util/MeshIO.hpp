// Copy from ipc

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Util {

bool readTetMesh(const std::string& filePath,
    Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
    Eigen::MatrixXi& F, bool findSurface = true);

bool readTetMesh_msh4(const std::string& filePath,
    Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
    Eigen::MatrixXi& F, bool findSurface = true);

void readNodeEle(const std::string& filePath,
    Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
    Eigen::MatrixXi& F);

}