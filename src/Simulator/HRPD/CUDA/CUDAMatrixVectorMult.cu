#include "CUDAMatrixVectorMult.hpp"
#include "helper_cuda.h"
#include <spdlog/spdlog.h>

__global__ void doubleToFloatMemCpyKernel(int n, int coord, double* source, float* target)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        target[i * 3 + coord] = source[i];
    }
}

__global__ void doubleToFloatMemCpyKernel(int n, int coord, float* source, float* target)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        target[i * 3 + coord] = source[i];
    }
}

__global__ void elementWiseMultiplyKernel(int n, double* a, double* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] * b[i];
    }
}

__global__ void elementWiseMultiplyKernel(int n, float* a, float* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] * b[i];
    }
}

void doubleToFloatDeviceCpy(int n, int coord, double* source, float* target)
{
    doubleToFloatMemCpyKernel<<<(n + 255) / 256, 256>>>(n, coord, source, target);
}

void doubleToFloatDeviceCpy(int n, int coord, float* source, float* target)
{
    doubleToFloatMemCpyKernel<<<(n + 255) / 256, 256>>>(n, coord, source, target);
}

void elementWiseMultiply(int n, double* a, double* b)
{
    elementWiseMultiplyKernel<<<(n + 255) / 256, 256>>>(n, a, b);
}

void elementWiseMultiply(int n, float* a, float* b)
{
    elementWiseMultiplyKernel<<<(n + 255) / 256, 256>>>(n, a, b);
}


__global__ void FloatMemCpyKernel(int n, float* source, float* target)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        target[i] = source[i];
    }
}

void FloatMemCpy(int n, float* source, float* target)
{
    FloatMemCpyKernel<<<(n + 255) / 256, 256>>>(n, source, target);
}


///////////////////////////////////////////////////////////////////////////////////////////
//  Struct CUDAMatrixVectorMultiplier
///////////////////////////////////////////////////////////////////////////////////////////
CUDAMatrixVectorMultiplier::CUDAMatrixVectorMultiplier(PD::PDMatrix& mat)
{
    if (cublasLibInitialized == false) {
        cublasCreate(&cublasLibHandle);
        cublasLibInitialized = true;
    }

    m_numRows = mat.rows();
    m_numCols = mat.cols();

    checkCudaErrors(cudaMalloc((void**)&m_cudaInVec, sizeof(PD::PDScalar) * m_numCols));
    checkCudaErrors(cudaMalloc((void**)&m_cudaMat, sizeof(PD::PDScalar) * m_numRows * m_numCols));
    checkCudaErrors(cudaMalloc((void**)&m_cudaOutVec, sizeof(PD::PDScalar) * m_numRows));
    // If the matrix has more columns then rows, we need to upload it transposed,
    // since cublasDgemv expects a matrix with more rows than columns...
    if (m_numRows < m_numCols) {
        PD::PDMatrix matT = mat.transpose();
        const void* matDataPointer = (const void*)matT.data();
        checkCudaErrors(cublasSetMatrix(m_numCols, m_numRows, sizeof(PD::PDScalar), matDataPointer, m_numCols, (void*)m_cudaMat, m_numCols));
    }
    else {
        const void* matDataPointer = (const void*)mat.data();
        checkCudaErrors(cublasSetMatrix(m_numRows, m_numCols, sizeof(PD::PDScalar), matDataPointer, m_numRows, (void*)m_cudaMat, m_numRows));
    }
}

CUDAMatrixVectorMultiplier::CUDAMatrixVectorMultiplier(PD::PDMatrix& mat, PD::PDVector& masses)
    : CUDAMatrixVectorMultiplier(mat)
{
    m_massesSize = masses.rows();
    checkCudaErrors(cudaMalloc((void**)&m_cudaMassesVec, sizeof(PD::PDScalar) * m_massesSize));
    checkCudaErrors(cublasSetVector(m_massesSize, sizeof(PD::PDScalar), masses.data(), 1, (void*)(m_cudaMassesVec), 1));
    cudaDeviceSynchronize();
}

CUDAMatrixVectorMultiplier::~CUDAMatrixVectorMultiplier()
{
    cudaFree(m_cudaMat);
    cudaFree(m_cudaInVec);
    cudaFree(m_cudaOutVec);
    if (!m_cudaMassesVec) cudaFree(m_cudaMassesVec);
}

void CUDAMatrixVectorMultiplier::mult(const void* inData, void* outData, PD::PDScalar& alpha, bool transpose, int coord, int cutoff)
{
    if (!transpose) {
        checkCudaErrors(cublasSetVector(m_numCols, sizeof(PD::PDScalar), inData, 1, (void*)(m_cudaInVec), 1));
        if (m_massesSize == m_numCols && m_cudaMassesVec) {
            elementWiseMultiply(m_numCols, m_cudaInVec, m_cudaMassesVec);
        }
    }
    else {
        checkCudaErrors(cublasSetVector(m_numRows, sizeof(PD::PDScalar), inData, 1, (void*)(m_cudaOutVec), 1));
        if (m_massesSize == m_numRows && m_cudaMassesVec) {
            elementWiseMultiply(m_numRows, m_cudaOutVec, m_cudaMassesVec);
        }
    }
    if (m_numRows < m_numCols) {
        if (!transpose) {
            // In this case a product with the untransposed matrix is desired, however, the matrix stored
            // on the GPU has been transposed before (since it had more columns than rows), so the operation
            // for Dgemv should be OP_T.
            // On the other hand the pointers to cudaInVec and cudaOutVec have the correct sizes (M and N
            // respectively) so that they appear in the normal order.
            // The reasoning for the other three cases below can be deduced from this example.
            checkCudaErrors(cublasDgemv(cublasLibHandle, CUBLAS_OP_T, m_numCols, m_numRows, &alpha, m_cudaMat, m_numCols, m_cudaInVec, 1, &cublasZero, m_cudaOutVec, 1));
        }
        else {
            checkCudaErrors(cublasDgemv(cublasLibHandle, CUBLAS_OP_N, m_numCols, m_numRows, &alpha, m_cudaMat, m_numCols, m_cudaOutVec, 1, &cublasZero, m_cudaInVec, 1));
        }
    }
    else {
        if (!transpose) {
            checkCudaErrors(cublasDgemv(cublasLibHandle, CUBLAS_OP_N, m_numRows, m_numCols, &alpha, m_cudaMat, m_numRows, m_cudaInVec, 1, &cublasZero, m_cudaOutVec, 1));
        }
        else {
            checkCudaErrors(cublasDgemv(cublasLibHandle, CUBLAS_OP_T, m_numRows, m_numCols, &alpha, m_cudaMat, m_numRows, m_cudaOutVec, 1, &cublasZero, m_cudaInVec, 1));
        }
    }

    if (outData) {
        if (!transpose) {
            checkCudaErrors(cudaMemcpy(outData, m_cudaOutVec, sizeof(PD::PDScalar) * m_numRows, cudaMemcpyDeviceToHost));
        }
        else {
            checkCudaErrors(cudaMemcpy(outData, m_cudaInVec, sizeof(PD::PDScalar) * m_numCols, cudaMemcpyDeviceToHost));
        }
    }

    // BUFFER MAP
    if (bufferInitifalized) {
        // if (!m_glArrayPtr) {
        //     // Hi-jack the buffer from OpenGL
        //     cudaGraphicsResource_t res;
        //     glBindBuffer(GL_ARRAY_BUFFER, m_glbufferId);
        //     checkCudaErrors(cudaGraphicsGLRegisterBuffer(&res, m_glbufferId, cudaGraphicsRegisterFlagsNone));
        //     checkCudaErrors(cudaGraphicsMapResources(1, &res, 0));
        //     size_t size;
        //     checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_glArrayPtr, &size, res));
        //     glBindBuffer(GL_ARRAY_BUFFER, 0);
        // }
        // if (m_glArrayPtr) {
        //     //Copy from m_cudaOutVec to m_glArrayPtr while casting from PDScalar to float
        //     int N = m_numRows;
        //     if (cutoff >= 0) {
        //         N = cutoff;
        //     }
        //     doubleToFloatDeviceCpy(N, coord, m_cudaOutVec, m_glArrayPtr);
        // }

        // Copy from m_cudaOutVec to m_glArrayPtr while casting from PDScalar to float
        int N = m_numRows;
        if (cutoff >= 0) {
            N = cutoff;
        }
        checkCudaErrors(cudaGraphicsMapResources(1, &m_res, 0));
        size_t size;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_glArrayPtr, &size, m_res));
        doubleToFloatDeviceCpy(N, coord, m_cudaOutVec, m_glArrayPtr);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGraphicsUnmapResources(1, &m_res, 0));
    }

    checkCudaErrors(cudaDeviceSynchronize());
}

void CUDAMatrixVectorMultiplier::initBufferMap(GLuint bufferId)
{
    spdlog::info(">>> CUDAMatrixVectorMultiplier::initBufferMap()");
    m_glbufferId = bufferId;
    glBindBuffer(GL_ARRAY_BUFFER, m_glbufferId);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_res, m_glbufferId, cudaGraphicsRegisterFlagsNone));
    bufferInitifalized = true;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    spdlog::info(">>> CUDAMatrixVectorMultiplier::initBufferMap() - After");
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Struct CUDASparseMatrixVectorMultiplier
///////////////////////////////////////////////////////////////////////////////////////////
CUDASparseMatrixVectorMultiplier::CUDASparseMatrixVectorMultiplier(PD::PDSparseMatrix& mat)
{
    spdlog::info(">>> CUDASparseMatrixVectorMultiplier Init - Before");
    checkCudaErrors(cusparseCreate(&m_handle));

    checkCudaErrors(cusparseCreateMatDescr(&m_descr));
    checkCudaErrors(cusparseSetMatType(m_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(m_descr, CUSPARSE_INDEX_BASE_ZERO));

    // Eigen sparse matrix deafult format is csc.
    auto& M_csc = mat;
    M_csc.makeCompressed();
    m_num_rows = M_csc.rows();
    m_num_cols = M_csc.cols();
    m_num_nonzeros = M_csc.nonZeros();

    checkCudaErrors(cudaMalloc(&m_M_values, sizeof(PD::PDScalar) * m_num_nonzeros));
    checkCudaErrors(cudaMalloc(&m_M_rowInd, sizeof(int) * m_num_nonzeros));
    checkCudaErrors(cudaMalloc(&m_M_colPtr, sizeof(int) * (m_num_cols + 1)));

    checkCudaErrors(cudaMemcpy(m_M_values, M_csc.valuePtr(), sizeof(PD::PDScalar) * m_num_nonzeros, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_M_rowInd, M_csc.innerIndexPtr(), sizeof(int) * m_num_nonzeros, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_M_colPtr, M_csc.outerIndexPtr(), sizeof(int) * (m_num_cols + 1), cudaMemcpyHostToDevice));

    // Manually create the CSC data for the sparse matrix
	// PD::PDScalar* entries = new PD::PDScalar[m_num_nonzeros];
	// int* rowInds = new int[m_num_nonzeros];
	// int* colPtr = new int[m_num_cols + 1];
	// unsigned int counter = 0;
	// colPtr[0] = 0;
	// for (int k = 0; k < mat.outerSize(); ++k) {
	// 	for (PD::PDSparseMatrix::InnerIterator it(mat, k); it; ++it)
	// 	{
	// 		entries[counter] = it.value();
	// 		rowInds[counter] = it.row();
	// 		counter++;
	// 	}
	// 	colPtr[k + 1] = counter;
	// }
    // checkCudaErrors(cudaMemcpy(m_M_values, entries, sizeof(PD::PDScalar) * m_num_nonzeros, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(m_M_rowInd, rowInds, sizeof(int) * m_num_nonzeros, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(m_M_colPtr, colPtr, sizeof(int) * (m_num_cols + 1), cudaMemcpyHostToDevice));

    checkCudaErrors(cusparseCreateCsc(&m_mat, m_num_rows, m_num_cols, m_num_nonzeros, m_M_colPtr,
        m_M_rowInd, m_M_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    checkCudaErrors(cudaMalloc(&m_V_in_dv, sizeof(PD::PDScalar) * m_num_cols));
    checkCudaErrors(cudaMalloc(&m_V_out_dv, sizeof(PD::PDScalar) * m_num_rows));

    checkCudaErrors(cusparseCreateDnVec(&m_vecIn, m_num_cols, m_V_in_dv, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&m_vecOut, m_num_rows, m_V_out_dv, CUDA_R_64F));

    alpha = 1.0;
    beta = 0.0;

    // Query buffer size
    checkCudaErrors(cusparseSpMV_bufferSize(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, m_mat, m_vecIn, &beta, m_vecOut, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    // Allocate buffer
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));

    spdlog::info(">>> CUDASparseMatrixVectorMultiplier Init - After");
}

CUDASparseMatrixVectorMultiplier::~CUDASparseMatrixVectorMultiplier()
{
    checkCudaErrors(cusparseDestroySpMat(m_mat));
    checkCudaErrors(cusparseDestroyDnVec(m_vecIn));
    checkCudaErrors(cusparseDestroyDnVec(m_vecOut));
    checkCudaErrors(cusparseDestroy(m_handle));

    checkCudaErrors(cudaFree(m_M_values));
    checkCudaErrors(cudaFree(m_M_rowInd));
    checkCudaErrors(cudaFree(m_M_colPtr));
    checkCudaErrors(cudaFree(m_V_in_dv));
    checkCudaErrors(cudaFree(m_V_out_dv));
}

void CUDASparseMatrixVectorMultiplier::mult(void* inData, void* outData, int coord, int cutoff)
{
    checkCudaErrors(cublasSetVector(m_num_cols, sizeof(PD::PDScalar), inData, 1, (void*)(m_V_in_dv), 1));
    checkCudaErrors(cudaMemcpy(m_V_in_dv, inData, sizeof(PD::PDScalar) * m_num_cols, cudaMemcpyHostToDevice));
    checkCudaErrors(cusparseDnVecSetValues(m_vecIn, m_V_in_dv));
    checkCudaErrors(cusparseSpMV(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, m_mat, m_vecIn, &beta, m_vecOut, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    // checkCudaErrors(cusparseDnVecGetValues(m_vecOut, (void**) & m_V_out_dv));
    checkCudaErrors(cudaMemcpy(outData, m_V_out_dv, sizeof(PD::PDScalar) * m_num_rows, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());
}

void CUDASparseMatrixVectorMultiplier::initBufferMap(GLuint bufferId)
{
    spdlog::info(">>> CUDASparseMatrixVectorMultiplier::initBufferMap()");
    m_glbufferId = bufferId;
    glBindBuffer(GL_ARRAY_BUFFER, m_glbufferId);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_res, m_glbufferId, cudaGraphicsRegisterFlagsNone));
    bufferInitifalized = true;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    spdlog::info(">>> CUDASparseMatrixVectorMultiplier::initBufferMap() - After");
}

void CUDASparseMatrixVectorMultiplier::bufferMap(void* inData, int coord, int cutoff)
{
    checkCudaErrors(cublasSetVector(m_num_rows, sizeof(PD::PDScalar), inData, 1, (void*)(m_V_out_dv), 1));
    checkCudaErrors(cudaMemcpy(m_V_out_dv, inData, sizeof(PD::PDScalar) * m_num_rows, cudaMemcpyHostToDevice));
    if (bufferInitifalized) {
        // Copy from outData to m_glArrayPtr while casting from PDScalar to float
        int N = m_num_rows;
        if (cutoff >= 0) {
            N = cutoff;
        }
        checkCudaErrors(cudaGraphicsMapResources(1, &m_res, 0));
        size_t size;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_glArrayPtr, &size, m_res));
        doubleToFloatDeviceCpy(N, coord, static_cast<PD::PDScalar*>(m_V_out_dv), m_glArrayPtr);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGraphicsUnmapResources(1, &m_res, 0));
    }
    checkCudaErrors(cudaDeviceSynchronize());
}