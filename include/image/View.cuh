/**
 * @file View.cuh
 * @brief Device-side view interface for image processing kernels
 * 
 * This is class-interface, created in order to work with device-oriented data
 * from kernels. This class should be passed into all of the kernels, which works
 * with image processing.
 *
 * All functions are `__device__` oriented with no stl sub-structures in
 * functions nor in class.
 */

#pragma once

#include "Pixel.cuh"

/**
 * @namespace device_view
 * @brief Namespace for device-side image viewing classes
 */
namespace device_view {

/**
 * @class View
 * @brief Device-side image view for CUDA kernels
 * 
 * This lightweight class provides a device-side interface to image data.
 * It stores only a pointer to device memory along with dimensions,
 * avoiding any STL containers that cannot be used on the GPU.
 * Used to pass image data to CUDA kernels efficiently.
 */
class View {
    pixel::RGB *content; ///< Device memory pointer for image data
    int width;           ///< Image width in pixels
    int height;          ///< Image height in pixels

public:
    /**
     * @brief Constructor for View
     * @param p Pointer to device memory containing pixel data
     * @param w Image width
     * @param h Image height
     */
    __host__ __device__ View(pixel::RGB *p = nullptr, int w = 0, int h = 0) : content(p), width(w), height(h) {}

    /**
     * @brief Set pixel color at specified coordinates
     * @param x X coordinate
     * @param y Y coordinate
     * @param color RGB color value to set
     */
    __device__ void set_pixel(int x, int y, pixel::RGB color) {
        if (x < width && y < height) {
            content[y * width + x] = color;
        }
    }

    /**
     * @brief Get pixel color at specified coordinates (no bounds checking)
     * @param x X coordinate
     * @param y Y coordinate
     * @return RGB pixel value
     */
    __device__ pixel::RGB get_pixel(int x, int y) const { return content[y * width + x]; }

    /**
     * @brief Safe pixel access with boundary checking and clamping
     * @param x X coordinate (will be clamped to valid range)
     * @param y Y coordinate (will be clamped to valid range)
     * @return RGB pixel value
     */
    __device__ pixel::RGB get_pixel_safe(int x, int y) const {
        if (x < 0) x = 0;
        if (x >= width) x = width - 1;
        if (y < 0) y = 0;
        if (y >= height) y = height - 1;
        return content[y * width + x];
    }

    /**
     * @brief Get image height
     * @return Image height in pixels
     */
    __device__ int get_height() const { return height; }

    /**
     * @brief Get image width
     * @return Image width in pixels
     */
    __device__ int get_width() const { return width; }
};
} // namespace device_view