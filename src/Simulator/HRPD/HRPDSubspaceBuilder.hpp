#pragma once

#include "Simulator/HRPD/HRPDTetMesh.hpp"
#include "UI/InteractState.hpp"
#include "Simulator/PDTypeDef.hpp"
#include <memory>

#ifdef PD_USE_CUDA
#include "CUDA/CUDAMatrixVectorMult.hpp"
#endif

#define USE_PCA false
/* If this is turned on, the basis from the skinning
   construction is created without any weights, the interpolation will
   also be done without the weights, and only after the subspace rhs
   vector has been found, the weights are applied. */
#define NO_WEIGHTS_IN_CONSTRUCTION true
#define MIN_OVERSAMPLING 20

extern UI::InteractState g_InteractState;

namespace PD {

struct SubspaceBuilder {
    std::shared_ptr<HRPDTetMesh> m_mesh;

    // For Mesh Sample
    STVDSampler m_sampler;

    // [N x N]
    PDSparseMatrix m_massMatrixForPos;
    PDVector m_massMatrixDiagForPos;
    // [3e x 3e]
    PDSparseMatrix m_massMatrixForProj;
    PDVector m_massMatrixDiagForProj;
    PDSparseMatrix m_weightMatrixForProj;
    PDVector m_weightMatrixDiagForProj;

    std::vector<std::vector<std::pair<unsigned int, unsigned int>>> m_tetsPerVertex;

    // Subspace intepolation matrices
    PDMatrix m_U; // For Position - [N x 4k]
    PDMatrix m_UT;
    PDMatrix m_V; // For Projection - [3e x (ks + 1)]
    PDSparseMatrix m_J; // For resolution-independent - [3kt x 3e]

    PDSparseMatrix m_U_sp;
    PDSparseMatrix m_UT_sp;
    PDSparseMatrix m_V_sp;

    // Assembly Matrix - [N x 3e]
    PDSparseMatrix m_ST;

    // Samples
    std::vector<unsigned int> m_posSamples;
    std::vector<unsigned int> m_projSamples;
    std::vector<unsigned int> m_constraintVertexSamples;
    std::vector<unsigned int> m_sampledConstraintInds;
    PDTets m_sampledTetsForUsedVs;

    // For online compuation
    PDMatrix ms_VTJT;       // [(ks + 1) x 3kt]
    PDMatrix ms_UTSTWiV;    // [4k x (ks + 1)]

    // Solver
    PDDenseSolver m_fittingSolver;

    // -- For subspace: Project fullspace to subspace
    // ---- Solve UTMU * x_{sub} = UTM * x_{full}
    // ---- Question: Why not solve UTU * x_{sub} = UT * x_{full} ?
    PDMatrix m_UTMU;
    PDSparseSolver m_subspaceSolverSparse;

    // -- For used subspace vertices
    std::vector<unsigned int> m_usedVertices;
    PDPositions m_positionsUsedVs;
    PDPositions m_velocitiesUsedVs;
    PDPositions m_velocitiesRepulsion;
    PDPositions m_velocitiesUsedVsRepulsion;
    PDPositions m_projectionsUsedVs;
    // ---- U'T [U_{used}.transpose()] [4k x N']
    PDMatrix m_U_used;
    PDMatrix m_UT_used;
    PDSparseMatrix m_U_used_sp;
    PDSparseMatrix m_UT_used_sp;
    // ---- Solve (UT_{used} * U_{used}) * x_{sub} = (UT_{used}) * x_{used}
    // ---- Get q_{sub} from q_{used} when q_{used} is changed, i.e. collision handling.
    // ---- But why not directly solve (UT * M * U) * x_{sub} = (UT * M) * q_{full} ?
    // PDDenseLLTSolver m_usedSubspaceSolverDense;
    PDSparseSolver m_usedSubspaceSolverSparse;

#ifdef PD_USE_CUDA
    CUDASparseMatrixVectorMultiplier* m_vPosGPUUpdater = nullptr;
    // CUDASparseMatrixVectorMultiplier* m_vsPosGPUUpdater = nullptr;
#endif

public:
    void Init(std::shared_ptr<HRPDTetMesh> mesh);

    PDMatrix SnapshotPCA(PDMatrix& Y, PDVector& masses);
    // For skinning space
    PDMatrix GetRadialBaseFunctions(std::vector<unsigned int>& samples, bool partitionOfOne, double r, double eps = -1., int numSmallSamples = -1, double smallSamplesRadius = 1.);
    // [N x (4k + 1)]
    PDMatrix CreateSkinningSpace(PDPositions& restPositions, PDMatrix& weights, unsigned int linesPerConstraint = 1);

public:
    void CreateSubspacePosition();
    void CreateSubspaceProjection();
    void CreateProjectionInterpolationMatrix();

    void InitProjection();
    void InitInterpolation();

    void ProjectFullspaceToSubspaceForPos(PDPositions& sub, PDPositions& full);
    void ProjectUsedFullspaceToSubspaceForPos(PDPositions& sub, PDPositions& usedfull);
    void InterpolateSubspaceToFullspaceForPos(PDPositions& posFull, PDPositions& posSub, bool usedVerticesOnly);

    void GetUsedP();
};

}