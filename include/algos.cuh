/**
 * @file algos.cuh
 * @brief CUDA kernel declarations for image processing algorithms
 */

#pragma once
#include "image/View.cuh"

/**
 * @brief CUDA kernel for computing column-wise prefix sum (integral image)
 * @param image Pointer to device view of the image
 */
__global__ void col_prefix_sum(device_view::View *image);

/**
 * @brief CUDA kernel for computing row-wise prefix sum (integral image)
 * @param image Pointer to device view of the image
 * @param N Number of rows to process
 */
__global__ void row_prefix_sum(device_view::View *image, int N);