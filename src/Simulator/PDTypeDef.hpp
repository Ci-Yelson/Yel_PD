#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifndef EIGEN_DONT_PARALLELIZE
#define PD_OMP_NUM_THREADS 8
#define PD_EIGEN_NUM_THREADS 4
#else
#define PD_OMP_NUM_THREADS 8
#endif

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define PD_PARALLEL_FOR DO_PRAGMA(omp parallel for num_threads(PD_OMP_NUM_THREADS))
#define PD_PARALLEL_ATOMIC DO_PRAGMA(omp atomic)

#define PD_CUTOFF (1e-10)
// Entries below this value will be removed from matrices when sparsifying them
#define PD_SPARSITY_CUTOFF (1e-12)

// Corrects mass of very small triangles to have a minimal mass
#define PD_MIN_MASS (1e-10f)

namespace PD {

typedef double PDScalar;
typedef int PDIndex;
template <int rows, int cols>
using PDMatrixF = Eigen::Matrix<PDScalar, rows, cols>;
typedef Eigen::Matrix<PDScalar, -1, 1> PDVector;
typedef Eigen::Matrix<PDScalar, 3, 1> PD3dVector;
typedef Eigen::Matrix<PDScalar, 2, 1> PD2dVector;
typedef Eigen::Matrix<PDScalar, -1, -1> PDMatrix;
typedef Eigen::Matrix<PDScalar, -1, 3> PDPositions;
typedef Eigen::SparseMatrix<PDScalar, Eigen::ColMajor> PDSparseMatrix;
typedef Eigen::SparseMatrix<PDScalar, Eigen::RowMajor> PDSparseMatrixRM;
typedef Eigen::Matrix<PDIndex, -1, 3> PDTriangles;
typedef Eigen::Matrix<PDIndex, -1, 4> PDTets;

typedef Eigen::Triplet<PDScalar, PDIndex> PDSparseMatrixTriplet;
typedef Eigen::Quaternion<PDScalar, Eigen::DontAlign> EigenQuaternion;

typedef Eigen::Matrix<PDScalar, 12, 12, 0, 12, 12> EigenMatrix12;
typedef Eigen::Matrix<PDScalar, 12, 1, 0, 12, 1> EigenVector12;
typedef Eigen::Matrix<PDScalar, 4, 4, 0, 4, 4> EigenMatrix4;
typedef Eigen::Matrix<PDScalar, 4, 1, 0, 4, 1> EigenVector4;
typedef Eigen::Matrix<PDScalar, 3, 3, 0, 3, 3> EigenMatrix3;
typedef Eigen::Matrix<PDScalar, 3, 1, 0, 3, 1> EigenVector3;
typedef Eigen::Matrix<PDScalar, 2, 2, 0, 2, 2> EigenMatrix2;
typedef Eigen::Matrix<PDScalar, 2, 1, 0, 2, 1> EigenVector2;
typedef Eigen::Matrix<PDScalar, -1, 3, 0, -1, 3> EigenMatrixx3;

typedef Eigen::LDLT<PDMatrix> PDDenseSolver;
typedef Eigen::LLT<PDMatrix> PDDenseLLTSolver;
typedef Eigen::SimplicialLDLT<PDSparseMatrix> PDSparseSolver;
typedef Eigen::SimplicialLLT<PDSparseMatrix> PDSparseLLTSolver;
typedef Eigen::ConjugateGradient<PDSparseMatrix> PDSparseCGSolver;

} // namespace PD