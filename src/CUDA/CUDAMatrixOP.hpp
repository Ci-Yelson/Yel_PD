#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "Simulator/PDTypeDef.hpp"

// OP: L = U.T * M * U
void CUDAMatrixUTMU(PD::PDMatrix &U, PD::PDSparseMatrix &M, PD::PDMatrix &L);

// OP: L = U.T * S.T * M * S * U
void MatrixUTSTMSU(PD::PDMatrix &U, PD::PDMatrix &S, PD::PDSparseMatrix &M, PD::PDMatrix &L);