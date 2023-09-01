#ifndef _YEL_CUDA_MATRIX_VECTOR_MULT_H_
#define _YEL_CUDA_MATRIX_VECTOR_MULT_H_

#pragma once

#include "Simulator/PDTypeDef.hpp"
#include "Util/Timer.hpp"

// For DIRECT BUFFER MAP
#ifndef __gl_h_
#ifndef __GL_H_
#include <glad/glad.h>
#endif
#endif

#include <iostream>
#include <cublas_v2.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "helper_cuda.h"

void doubleToFloatDeviceCpy(int n, int coord, double* source, float* target);
void elementWiseMultiply(int n, double* a, double* b);
void doubleToFloatDeviceCpy(int n, int coord, float* source, float* target);
void elementWiseMultiply(int n, float* a, float* b);

void FloatMemCpy(int n, float* source, float* target);

///////////////////////////////////////////////////////////////////////////////////////////
//  Struct CUDAMatrixVectorMultiplier
///////////////////////////////////////////////////////////////////////////////////////////
struct CUDAMatrixVectorMultiplier {
    inline static bool cublasLibInitialized = false;
    inline static cublasHandle_t cublasLibHandle = 0;
    inline static PD::PDScalar cublasZero = 0.0;

    unsigned int m_numCols, m_numRows;

    bool bufferInitifalized = false;
    GLuint m_glbufferId;
    float* m_glArrayPtr = nullptr;
    cudaGraphicsResource_t m_res;

    int m_massesSize = 0;

    PD::PDScalar* m_cudaMat = nullptr;
    PD::PDScalar* m_cudaInVec = nullptr;
    PD::PDScalar* m_cudaOutVec = nullptr;
    PD::PDScalar* m_cudaMassesVec = nullptr;

    Util::StopWatch m_multTime;
    Util::StopWatch m_getTime;
    Util::StopWatch m_setTime;

public:
    CUDAMatrixVectorMultiplier(PD::PDMatrix& mat);
    CUDAMatrixVectorMultiplier(PD::PDMatrix& mat, PD::PDVector& masses);
    ~CUDAMatrixVectorMultiplier();
    void mult(const void* inData, void* outData, PD::PDScalar& alpha, bool transpose = false, int coord = 0, int cutoff = -1);
    void initBufferMap(GLuint bufferId);
};

///////////////////////////////////////////////////////////////////////////////////////////
//  Struct CUDASparseMatrixVectorMultiplier
///////////////////////////////////////////////////////////////////////////////////////////
struct CUDASparseMatrixVectorMultiplier {
	cusparseHandle_t m_handle;
    cusparseMatDescr_t m_descr;
    cusparseSpMatDescr_t m_mat;
    cusparseDnVecDescr_t m_vecIn;
    cusparseDnVecDescr_t m_vecOut;
    double alpha;
    double beta;
    size_t bufferSize;
    void* buffer;

    int m_num_rows, m_num_cols, m_num_nonzeros;

    void* m_M_values = nullptr;
    void* m_M_rowInd = nullptr;
    void* m_M_colPtr = nullptr;

    void* m_V_in_dv = nullptr;
    void* m_V_out_dv = nullptr;

    bool bufferInitifalized = false;
    GLuint m_glbufferId;
    float* m_glArrayPtr = nullptr;
    cudaGraphicsResource_t m_res;

public:
    CUDASparseMatrixVectorMultiplier(PD::PDSparseMatrix& mat);
    ~CUDASparseMatrixVectorMultiplier();
    void mult(void* inData, void* outData, int coord = 0, int cutoff = -1);
    void initBufferMap(GLuint bufferId);
    void bufferMap(void* inData, int coord = 0, int cutoff = -1);
};

struct CUDABufferMapping {
    bool bufferInitifalized = false;
    GLuint m_glbufferId = 0;
    float* m_glArrayPtr = nullptr;
    cudaGraphicsResource_t m_res = 0;

    void* m_cudaVec = nullptr;

public:
    CUDABufferMapping(){}
    ~CUDABufferMapping() {
        if (m_cudaVec) checkCudaErrors(cudaFree(m_cudaVec));
    }

    void initBufferMap(int size, GLuint bufferId)
    {
        if (bufferInitifalized) return;
        spdlog::info(">>> CUDASparseMatrixVectorMultiplier::initBufferMap() - size = {}, bufferId = {}", size, bufferId);
        checkCudaErrors(cudaMalloc(&m_cudaVec, size));
        m_glbufferId = bufferId;
        glBindBuffer(GL_ARRAY_BUFFER, m_glbufferId);
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_res, m_glbufferId, cudaGraphicsRegisterFlagsNone));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        bufferInitifalized = true;
        spdlog::info(">>> CUDASparseMatrixVectorMultiplier::initBufferMap() - After");
    }
    void bufferMap(float* inData, int length, cudaMemcpyKind cuMemcpykind = cudaMemcpyHostToDevice)
    {
        if (bufferInitifalized) {
            spdlog::info(">>> CUDASparseMatrixVectorMultiplier::bufferMap() - length = {}", length);
            checkCudaErrors(cudaMemcpy(m_cudaVec, inData, sizeof(float) * length, cuMemcpykind));

            // Copy from inData to m_glArrayPtr while casting from PDScalar to float
            checkCudaErrors(cudaGraphicsMapResources(1, &m_res, 0));
            size_t size;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_glArrayPtr, &size, m_res));
            FloatMemCpy(length, static_cast<float*>(m_cudaVec), m_glArrayPtr);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaGraphicsUnmapResources(1, &m_res, 0));
        }
    }
};

#endif