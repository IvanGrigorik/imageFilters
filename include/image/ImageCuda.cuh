#pragma once
// Cuda includes
#include "View.cuh"
#include "image/Image.cuh"

// Cuda includes
#include <curand.h>
#include <curand_kernel.h>


// CUDA kernel for box blurring
__global__ void box_blur_kernel(device_view::View image, device_view::View output, int boxSize);

// CUDA kernels for integral image-based box blur
__global__ void compute_row_prefix_sum(device_view::View input, device_view::View output);
__global__ void compute_col_prefix_sum(device_view::View input, device_view::View output);
__global__ void integral_box_blur_kernel(device_view::View integral, device_view::View output, int boxSize);

// CUDA kernels for Gaussian blur (separable approach)
__global__ void gaussian_blur_horizontal(device_view::View input, device_view::View output, float *kernel, int kernelRadius);
__global__ void gaussian_blur_vertical(device_view::View input, device_view::View output, float *kernel, int kernelRadius);
