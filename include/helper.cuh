#pragma once
#include <cuda_runtime.h>

// File with the helper functions, macros and constants

// Error checking macros
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK_RETURN(call)                                                                                        \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                         \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK_RETURN_VAL(call, val)                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                         \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

// Macro to select suitable function in function calls
#ifdef __CUDA_ARCH__
#define HD_SELECT(host_func, device_func) device_func
#else
#define HD_SELECT(host_func, device_func) host_func
#endif

// Max threads per block and max blocks
#define MAX_TPB 1'024
#define MAX_BLOCKS 65'535