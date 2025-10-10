#pragma once
// Cuda includes
#include "View.cuh"
#include "image/Image.cuh"

// Cuda includes
#include <curand.h>
#include <curand_kernel.h>


// CUDA kernel for box blurring
__global__ void box_blur_kernel(device_view::View image);

// // CUDA kernel for filling image with solid color
// __global__ void fill_image_plain(device_view::View image, pixel::RGB color);

// // CUDA kernel for randomly filling the image
// __global__ void fill_image_random(device_view::View image, unsigned long long seed);

// // CUDA kernel for filling image with gradient
// __global__ void fill_image_gradient(device_view::View image);