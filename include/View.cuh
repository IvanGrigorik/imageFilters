#pragma once

/*

  This is class-interface, created in order to work with device-oriented data
  from kernels. This class should be passed into all of the kernels, which works
  with image processing.

  All functions are `__device__` oriented with no stl sub-structures in
  functions nor in class.
*/

#include "Pixel.cuh"

namespace device_view {
class View {
    pixel::RGB *content; // Device memory for image data (just pointer, not a copy)
    int width;
    int height;

public:
    __host__ __device__ View(pixel::RGB *p = nullptr, int w = 0, int h = 0) : content(p), width(w), height(h) {}

    __device__ void set_pixel(int x, int y, pixel::RGB color) {
        if (x < width && y < height) {
            content[y * width + x] = color;
        }
    }

    __device__ pixel::RGB get_pixel(int x, int y) const { return content[y * width + x]; }

    __device__ int get_height() { return height; }

    __device__ int get_width() { return width; }
};
} // namespace device_view