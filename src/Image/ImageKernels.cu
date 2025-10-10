// Cuda includes
#include "Pixel.cuh"
#include "View.cuh"
#include "image/Image.cuh"

// Cuda includes
#include <curand.h>
#include <curand_kernel.h>

__global__ void fill_image_plain(device_view::View image, pixel::RGB color) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= image.get_width() || y >= image.get_height()) {
        return;
    }
    image.set_pixel(x, y, color);
}

__global__ void fill_image_random(device_view::View image, unsigned long long seed) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= image.get_width() || y >= image.get_height()) {
        return;
    }
    int idx = y * image.get_width() + x;
    curandState localState;
    curand_init(seed, idx, 0, &localState);

    pixel::RGB p(curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState));
    image.set_pixel(x, y, p);
}

__global__ void fill_image_gradient(device_view::View image) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= image.get_width() || y >= image.get_height()) {
        return;
    }
    auto r = double(x) / (image.get_width() - 1);
    auto g = double(y) / (image.get_height() - 1);
    auto b = 0.0;
    image.set_pixel(x, y, {r, g, b});
}