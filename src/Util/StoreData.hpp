#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <fstream>
#include <spdlog/spdlog.h>

namespace Util {

template <class T>
void storeData(std::vector<T>& vc, std::string url, bool active = false, bool storeFull = false)
{
    if (!active) return;
    spdlog::info("> Store File: {}", url);
    std::ofstream f;
    f.open(url);
    if (storeFull) {
        for (auto v : vc) f << v << "\n";
    }
    else {
        for (int i = 0; i < std::min(100, int(vc.size())); i++) f << vc[i] << "\n";
    }
    f.close();
};

inline void storeData(const Eigen::MatrixXd& M, std::string url, bool active = false, bool storeFull = false)
{
    if (!active) return;
    spdlog::info("> Store File: {}", url);
    std::ofstream f;
    f.open(url);
    f.setf(std::ios::fixed);
    f.setf(std::ios::showpoint);
    f.precision(8);
    f << M.rows() << " " << M.cols() << "\n";
    if (storeFull) {
        f << M << "\n";
    }
    else {
        f << M.block(0, 0, std::min(100, int(M.rows())), std::min(100, int(M.cols()))) << "\n";
    }
    f.close();
};

inline void storeData(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& tris, std::string url, bool active = false, bool storeFull = false)
{
    if (!active) return;
    spdlog::info("> Store File: {}", url);
    std::ofstream objFile;
    objFile.open(url);
    objFile.setf(std::ios::fixed);
    objFile.setf(std::ios::showpoint);
    objFile.precision(8);
    for (int v = 0; v < pos.rows(); v++) {
        objFile << "v " << pos(v, 0) << " " << pos(v, 1) << " " << pos(v, 2) << "\n";
    }
    objFile << "\n";
    for (int f = 0; f < tris.rows(); f++) {
        objFile << "f  " << (tris(f, 0) + 1) << " " << (tris(f, 1) + 1) << " " << (tris(f, 2) + 1) << "\n";
    }
    objFile.close();
}

inline void ExportToObj(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& tris, std::string url)
{
    spdlog::info("> Store File: {}", url);
    std::ofstream objFile;
    objFile.open(url);
    objFile.setf(std::ios::fixed);
    objFile.setf(std::ios::showpoint);
    objFile.precision(8);
    for (int v = 0; v < pos.rows(); v++) {
        objFile << "v " << pos(v, 0) << " " << pos(v, 1) << " " << pos(v, 2) << "\n";
    }
    objFile << "\n";
    for (int f = 0; f < tris.rows(); f++) {
        objFile << "f  " << (tris(f, 0) + 1) << " " << (tris(f, 1) + 1) << " " << (tris(f, 2) + 1) << "\n";
    }
    objFile.close();
}
};