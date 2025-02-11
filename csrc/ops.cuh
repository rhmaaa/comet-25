#ifndef ops_H
#define ops_H


#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <vector>
#include <functional>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>




#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

#define THREADS_PER_BLOCKS (512)

#define CHECK_CUSPARSE(value) {                      \
  cusparseStatus_t _m_cudaStat = value;                    \
  if (_m_cudaStat != CUSPARSE_STATUS_SUCCESS) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cusparseGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }


#define THREADS_PER_BLOCKS (512)


inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline int checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        //throw std::logic_error("cuBLAS API failed");
        return 1;
    }
    return 0;
}



class Context
{
    public:
				cublasHandle_t m_handle;

				Context()
				{
					cublasHandle_t handle;
					cublasCreate_v2(&handle);
					m_handle = handle;
				}

};

class ContextLt
{
    public:
				cublasLtHandle_t m_handle;

				ContextLt()
				{
					cublasLtHandle_t handle;
					cublasLtCreate(&handle);
					m_handle = handle;
				}

};

