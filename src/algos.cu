// Local includes
#include "Pixel.cuh"
#include "View.cuh"
#include "algos.cuh"
#include "image/Image.cuh"

// CUDA includes
#include <cuda_runtime.h>

using namespace pixel;

// Blelloch scan for rows
__global__ void row_prefix_sum(device_view::View *image, int N) {
    int y = blockIdx.x;    // one block per row
    int row = threadIdx.x; // one thread per pixel in row
    int width = image->get_width();

    extern __shared__ pixel::RGB temp[]; // size = width

    if (row >= width) {
        return;
    }

    temp[row] = image->get_pixel(row, y);
    __syncthreads();

    // Up-sweep
    for (int d = 1; d < N; d *= 2) {
        __syncthreads();
        int ai = (row + 1) * 2 * d - 1;
        int bi = (row + 1) * 2 * d - 1 + d;
        __syncthreads();
        int ai = (row + 1) * 2 * d - 1;
        int bi = (row + 1) * 2 * d - 1 + d;
        if (bi < width) {
            temp[bi] = temp[bi] + temp[ai];
        } else {
            break;
        }
    }

    // Down-sweep
    if (row == 0) {
        temp[width - 1] = {0, 0, 0};
    }
    // Down-sweep
    for (int d = width / 2; d >= 1; d /= 2) {
        __syncthreads();
        int ai = (row + 1) * 2 * d - 1;
        int bi = (row + 1) * 2 * d - 1 + d;
        if (bi < width) {
            pixel::RGB t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = temp[bi] + t;
        }
    }
    __syncthreads();

    // Write back
    if (row < width) {
        image->set_pixel(row, y, temp[row]);
    }
}

// Blelloch scan for columns
__global__ void col_prefix_sum(device_view::View *image) {
    int x = blockIdx.x;
    int tid = threadIdx.x;
    int height = image->get_height();

    extern __shared__ pixel::RGB temp[]; // size = height

    if (tid >= height) {
        return;
    }

    temp[tid] = image->get_pixel(x, tid);
    __syncthreads();

    // Up-sweep
    for (int d = 1; d < height; d *= 2) {
        __syncthreads();
        int ai = (tid + 1) * 2 * d - 1;
        int bi = (tid + 1) * 2 * d - 1 + d;
        if (bi < height)
            temp[bi] = temp[bi] + temp[ai];
    }

    // Down-sweep
    if (tid == 0) {
        temp[height - 1] = {0, 0, 0};
    }
    for (int d = height / 2; d >= 1; d /= 2) {
        __syncthreads();
        int ai = (tid + 1) * 2 * d - 1;
        int bi = (tid + 1) * 2 * d - 1 + d;
        if (bi < height) {
            pixel::RGB t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = temp[bi] + t;
        }
    }
    __syncthreads();

    if (tid < height) {
        image->set_pixel(x, tid, temp[tid]);
    }
}
