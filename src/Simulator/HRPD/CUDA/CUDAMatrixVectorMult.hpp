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

void doubleToFloatDeviceCpy(int n, int coord, double* source, float* target);
void elementWiseMultiply(int n, double* a, double* b);
void doubleToFloatDeviceCpy(int n, int coord, float* source, float* target);
void elementWiseMultiply(int n, float* a, float* b);

///////////////////////////////////////////////////////////////////////////////////////////
//  Struct
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

struct CUDASparseMatrixVectorMultiplier {
	inline static bool cusparseLibInitialized = false;
    inline static cusparseHandle_t cusparseLibHandle = 0;
	cusparseMatDescr_t m_desc = 0;

	unsigned int m_numCols, m_numRows;
	
    int m_nnz;
	PD::PDScalar* m_cudaInVec = nullptr;
	PD::PDScalar* m_cudaOutVec = nullptr;
	PD::PDScalar* m_cudaMatData = nullptr;

	double m_alpha;
	double m_zero;

	int* m_cudaColPtr = nullptr;
	int* m_cudaRowInd = nullptr;

public:
    CUDASparseMatrixVectorMultiplier(PD::PDSparseMatrix& mat);
    void mult(const void* inData, void* outData, PD::PDScalar& alpha);
};

#endif