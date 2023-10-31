#pragma once

#include <Eigen/Dense>

#include <atomic>
#include <fstream>
#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>

namespace UI {
struct DraggingState {
    bool isDragging{ false };
    int vertex{ -1 };
    float forceStrength = 0.1;
    Eigen::RowVector3d prevMouse;
    Eigen::RowVector3d force;
};

// For visulize data
// enum ShowDataType {
//     DISPLACEMENT,
//     INNER_FORCE_INTERPOL,
//     PD_ENERGY,
//     STVK_ENERGY
// };

struct InteractState {
    std::string meshURL = "models/armadillo.obj";
    double timeStep = 0.033; // 1/30 s.
    int substeps = 1;
    int numIterations = 10;

    std::string simulatorType = "HRPD";

    bool isBufferMapping = false;
    bool isUseFullspaceVerticesForCollisionHandling = false; // Seems not work properly

    double dHat = 0.005;

    // For Newton
    struct NewtonParams {
        double stifness_mu = 80;
        double stifness_lambda = 20;

        double gravityConstant = 1;

        // Line search
        bool ls_enable_line_search = true;
        bool ls_enable_exact_search = false;
        double ls_step_size = 1.0;
        double ls_alpha = 0.03;
        double ls_beta = 0.5;
    } newtonParams;

    // For HRPD
    struct HRPDParams {
        int numberSamplesForVertexPosSubspace = 150; // [k]
        double radiusMultiplierForPosSubspace = 1.1;
        int numberSamplesForVertexProjSubspace = 120; // [ks]
        double radiusMultiplierForProjSubspace = 2.2;
        int numberSampledConstraints = 1000; // [kt]

        double massPerUnitArea = 1.0;
        double gravityConstant = 0.0007;

        // --------- Physics Params ---------
        float wi = 0.0051;
        float sigmaMin = 1.00f;
        float sigmaMax = 1.00f;
    } hrpdParams;

    // For QNPD
    struct QNPDParams {
        std::string materialType = "MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG";
        std::string solverType = "SOLVER_TYPE_DIRECT_LLT";

        std::string integration_method = "INTEGRATION_IMPLICIT_EULER";
        std::string optimization_method = "OPTIMIZATION_METHOD_LBFGS";
        int iterative_solver_max_iteration = 10;

        double stiffness_attachment = 120;
        double stiffness_stretch = 80;      // mu
        double stiffness_bending = 20;      // lambda
        double stiffness_kappa = 100;       // kappa
        double stiffness_laplacian = 2 * stiffness_stretch + stiffness_bending;
        double damping_coefficient = 0.001;

        double gravityConstant = 1;

        int lbfgs_m = 5;

        bool ls_enable_line_search = true;
        bool ls_enable_exact_search = false;
        double ls_step_size = 1.0;
        double ls_alpha = 0.03;
        double ls_beta = 0.5;
    } qnpdParams;

    // For multi threads
    bool isMultiThreads = false;
    struct MultiThreadsParams {
        bool isInterpolationFrame = false;
        int interpolationFrameSize{ 12 };
        int maxInterpolationBufferSize{ 10 };
        std::atomic_int bufferFrameCnt{ 0 }, updateFrameCnt{ 0 };
    } multiThreadsParams;

    // --------- User Interaction ---------
    DraggingState draggingState;
    bool isFloorActive = true;
    double floorHeight = 0.0;
    bool isGravityActive = true;
    bool isSingleStep = false;

    // For visulize data
    // double maxDispDif = 100.f;
    double maxDispDif = 0.f;
    int showDataType{ 0 };
    bool isUniformShowData = true;
    bool isShowColors = true;
    bool isShowSampledVertices = false;

    inline void load(std::string filepath)
    {
        std::ifstream f(filepath);
        nlohmann::json data = nlohmann::json::parse(f);

        meshURL = data["meshURL"].get<std::string>();
        simulatorType = data["simulatorType"].get<std::string>();

        timeStep = data["timeStep"].get<double>();
        numIterations = data["numIterations"].get<int>();

        if (simulatorType == "HRPD") {
            hrpdParams.numberSamplesForVertexPosSubspace = data["numberSamplesForVertexPosSubspace"].get<int>();
            hrpdParams.radiusMultiplierForPosSubspace = data["radiusMultiplierForPosSubspace"].get<double>();
            hrpdParams.numberSamplesForVertexProjSubspace = data["numberSamplesForVertexProjSubspace"].get<int>();
            hrpdParams.radiusMultiplierForProjSubspace = data["radiusMultiplierForProjSubspace"].get<double>();
            hrpdParams.numberSampledConstraints = data["numberSampledConstraints"].get<int>();

            hrpdParams.massPerUnitArea = data["massPerUnitArea"].get<double>();
            hrpdParams.gravityConstant = data["gravityConstant"].get<double>();

            hrpdParams.wi = data["wi"].get<double>();
            hrpdParams.sigmaMax = data["sigmaMax"].get<double>();
            hrpdParams.sigmaMin = data["sigmaMin"].get<double>();
        }
        // ...

        if (simulatorType == "QNPD") {
            if (data.contains("QNPDParams")) {
                nlohmann::json j = data.at("QNPDParams").get<nlohmann::json>();
                auto& params = qnpdParams;
                params.stiffness_attachment = j.at("stiffness_attachment").get<double>();
                params.stiffness_stretch = j.at("stiffness_stretch").get<double>();
                params.stiffness_bending = j.at("stiffness_bending").get<double>();
                params.stiffness_kappa = j.at("stiffness_kappa").get<double>();

                params.gravityConstant = j.at("gravityConstant").get<double>();

                params.ls_enable_line_search = j.at("ls_enable_line_search").get<bool>();
            }
        }

        spdlog::info(">>> After load file: {}", filepath);
    }
};
} // namespace UI