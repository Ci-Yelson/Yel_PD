#include "CUDAMatrixOP.hpp"
#include "helper_cuda.h"
#include <cusparse.h>

#include <spdlog/spdlog.h>

// Now is only support `double`
void CUDAMatrixUTMU(PD::PDMatrix& U, PD::PDSparseMatrix& M, PD::PDMatrix& L)
{
    L.resize(U.cols(), U.cols());
    // Create CSC data
    int _nnz = M.nonZeros();
    std::vector<PD::PDScalar> entries(_nnz);
    std::vector<int> rowInds(_nnz);
    std::vector<int> colPtr(M.cols() + 1);
    int count = 0;
    colPtr[0] = 0;
    for (int k = 0; k < M.outerSize(); k++) {
        for (PD::PDSparseMatrix::InnerIterator it(M, k); it; ++it) {
            entries[count] = it.value();
            rowInds[count] = it.row();
            count++;
        }
        colPtr[k + 1] = count;
    }

    // Upload to GPU
    // - For SparseMatrix M
    PD::PDScalar* d_entries;
    int* d_rowInds;
    int* d_colPtr;
    checkCudaErrors(cudaMalloc((void**)&d_entries, _nnz * sizeof(PD::PDScalar)));
    checkCudaErrors(cudaMalloc((void**)&d_rowInds, _nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_colPtr, (M.cols() + 1) * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_entries, entries.data(), _nnz * sizeof(PD::PDScalar), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rowInds, rowInds.data(), _nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colPtr, colPtr.data(), (M.cols() + 1) * sizeof(int), cudaMemcpyHostToDevice));
    // For DenseMatrix U
    PD::PDScalar* d_U;
    int U_N = U.rows();
    int U_M = U.cols();
    checkCudaErrors(cudaMalloc((void**)&d_U, U.size() * sizeof(PD::PDScalar)));
    checkCudaErrors(cudaMemcpy(d_U, U.data(), U.size() * sizeof(PD::PDScalar), cudaMemcpyHostToDevice));
    // For DenseMatrix C
    PD::PDScalar* d_C;
    checkCudaErrors(cudaMalloc((void**)&d_C, (M.rows() * U.cols()) * sizeof(PD::PDScalar)));
    // For Result DenseMatrix L
    PD::PDScalar* d_L;
    checkCudaErrors(cudaMalloc((void**)&d_L, L.size() * sizeof(PD::PDScalar)));

    // Create Descripters
    cusparseSpMatDescr_t matA = 0;
    cusparseDnMatDescr_t matB = 0;
    cusparseDnMatDescr_t matC = 0;
    cusparseCreateCsc(&matA, M.rows(), M.cols(), _nnz, d_colPtr, d_rowInds,
        d_entries, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnMat(&matB, U.rows(), U.cols(), U.rows(), d_U, CUDA_R_64F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, M.rows(), U.cols(), M.rows(), d_C, CUDA_R_64F, CUSPARSE_ORDER_COL);

    //  - C = M * U <-> [M.rows(), U_M] = [M.rows(), M_M] * [M_M, U_M]
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    PD::PDScalar alpha = 1.0f;
    PD::PDScalar beta = 0.0f;
    size_t bufferSize;
    void* buffer = NULL;
    cusparseSpMM_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
        matB, &beta, matC, CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));
    auto state = cusparseSpMM(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
        &beta, matC, CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG1, buffer);
    // - L = U.transpose() * C <-> [U.cols(), U.cols()] = [U.cols(), N] * [N,
    // U.cols()]
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, U.cols(), U.cols(),
        M.rows(), &alpha, d_U, U.outerStride(), d_C, M.rows(), &beta,
        d_L, U.cols());

    checkCudaErrors(cudaMemcpy(L.data(), d_L, L.size() * sizeof(PD::PDScalar), cudaMemcpyDeviceToHost));

    // Free
    checkCudaErrors(cudaFree(d_entries));
    checkCudaErrors(cudaFree(d_rowInds));
    checkCudaErrors(cudaFree(d_colPtr));
    checkCudaErrors(cudaFree(d_U));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_L));
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
}

// Now is only support `double`
void MatrixUTSTMSU(PD::PDMatrix &U, PD::PDSparseMatrix &S, PD::PDSparseMatrix &M, PD::PDMatrix &L)
{
    spdlog::info(">>> TODO !!!");
    return ;
    // ST * M
    // Eigen sparse matrix deafult format is csc.
    // Malloc for M
    void* M_values = nullptr;
    void* M_rowInd = nullptr;
    void* M_colPtr = nullptr;
    checkCudaErrors(cudaMalloc(&M_values, sizeof(PD::PDScalar) * M.nonZeros()));
    checkCudaErrors(cudaMalloc(&M_rowInd, sizeof(int) * M.nonZeros()));
    checkCudaErrors(cudaMalloc(&M_colPtr, sizeof(int) * (M.cols() + 1)));
    checkCudaErrors(cudaMemcpy(M_values, M.valuePtr(), sizeof(PD::PDScalar) * M.nonZeros(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(M_rowInd, M.innerIndexPtr(), sizeof(int) * M.nonZeros(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(M_colPtr, M.outerIndexPtr(), sizeof(int) * (M.cols() + 1), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t M_mat = 0;
    cusparseDnVecDescr_t M_vecX = 0;
    cusparseDnVecDescr_t M_vecY = 0;
    checkCudaErrors(cusparseCreateCsc(&M_mat, M.rows(), M.cols(), M.nonZeros(), M_colPtr,
        M_rowInd, M_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Malloc for S
    void* S_values = nullptr;
    void* S_rowInd = nullptr;
    void* S_colPtr = nullptr;
    checkCudaErrors(cudaMalloc(&S_values, sizeof(PD::PDScalar) * S.nonZeros()));
    checkCudaErrors(cudaMalloc(&S_rowInd, sizeof(int) * S.nonZeros()));
    checkCudaErrors(cudaMalloc(&S_colPtr, sizeof(int) * (S.cols() + 1)));
    checkCudaErrors(cudaMemcpy(S_values, S.valuePtr(), sizeof(PD::PDScalar) * S.nonZeros(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(S_rowInd, S.innerIndexPtr(), sizeof(int) * S.nonZeros(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(S_colPtr, S.outerIndexPtr(), sizeof(int) * (S.cols() + 1), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t S_mat = 0;
    cusparseDnVecDescr_t S_vecX = 0;
    cusparseDnVecDescr_t S_vecY = 0;
    checkCudaErrors(cusparseCreateCsc(&S_mat, M.rows(), M.cols(), M.nonZeros(), S_colPtr,
        S_rowInd, S_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    // todo...
    // cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2, void *externalBuffer2)
}