/**
 * @file Pixel.cuh
 * @brief Wrapper class for pixels of different color formats
 */

#pragma once
#include <stdio.h>

// Declare the image types in here (do not need to know anything except
// declaration)
namespace render {
class ImageBase;
}

/**
 * @namespace pixel
 * @brief Namespace for different color formats
 */
namespace pixel {

/**
 * @class RGB
 * @brief Classic RGB pixel format supporting both host and device operations
 * 
 * This class represents a pixel in RGB color space. It can be used both on
 * the host (CPU) and device (GPU) thanks to __host__ __device__ annotations.
 * Supports various arithmetic operations for color manipulation.
 */
class RGB {
private:
    int R; ///< Red channel value
    int G; ///< Green channel value
    int B; ///< Blue channel value

public:
    /**
     * @brief Default constructor
     */
    __host__ __device__ RGB() {};
    
    /**
     * @brief Constructor with integer RGB values
     * @param r Red channel (0-255)
     * @param g Green channel (0-255)
     * @param b Blue channel (0-255)
     */
    __host__ __device__ RGB(int r, int g, int b) : R(r), G(g), B(b) {};

    /**
     * @brief Constructor with normalized double RGB values
     * @param r Red channel (0.0-1.0)
     * @param g Green channel (0.0-1.0)
     * @param b Blue channel (0.0-1.0)
     */
    __host__ __device__ RGB(double r, double g, double b)
        : R(int(r * 255.999)), G(int(g * 255.999)), B(int(b * 255.999)) {};

    /**
     * @brief Constructor with normalized float RGB values
     * @param r Red channel (0.0-1.0)
     * @param g Green channel (0.0-1.0)
     * @param b Blue channel (0.0-1.0)
     */
    __host__ __device__ RGB(float r, float g, float b)
        : R(int(r * 255.9999)), G(int(g * 255.9999)), B(int(b * 255.9999)) {};

    // Befriend all of the pixel classes
    friend class render::ImageBase;

    /**
     * @brief Addition operator for color addition
     * @param c RGB color to add
     * @return Result of color addition
     */
    __host__ __device__ RGB operator+(const RGB &c) {
        R += c.R;
        G += c.G;
        B += c.B;
        return *this;
    }

    /**
     * @brief Subtraction operator for color subtraction
     * @param c RGB color to subtract
     * @return Result of color subtraction
     */
    __host__ __device__ RGB operator-(const RGB &c) {
        R -= c.R;
        G -= c.G;
        B -= c.B;
        return *this;
    }

    /**
     * @brief Scalar multiplication operator
     * @param t Scalar multiplier
     * @return Result of scalar multiplication
     */
    __host__ __device__ RGB operator*(double t) {
        R *= t;
        G *= t;
        B *= t;
        return *this;
    }

    /**
     * @brief Scalar division operator
     * @param t Scalar divisor
     * @return Result of scalar division
     */
    __host__ __device__ RGB operator/(double t) {
        if (t != 0) {
            R /= t;
            G /= t;
            B /= t;
        }
        return *this;
    }

    /**
     * @brief Modulo operator for each channel
     * @param t Modulo value
     * @return Result of modulo operation
     */
    __host__ __device__ RGB operator%(int t) {
        R = R % t;
        G = G % t;
        B = B % t;
        return *this;
    }

    /**
     * @brief Get a copy of this RGB value
     * @return Copy of this RGB pixel
     */
    __host__ __device__ RGB get() const { return *this; }

    /**
     * @brief Get red channel value (clamped to 0-255 range)
     * @return Red channel value
     */
    __host__ __device__ int getR() const { return R % 256; }

    /**
     * @brief Get green channel value (clamped to 0-255 range)
     * @return Green channel value
     */
    __host__ __device__ int getG() const { return G % 256; }

    /**
     * @brief Get blue channel value (clamped to 0-255 range)
     * @return Blue channel value
     */
    __host__ __device__ int getB() const { return B % 256; }
    
    /**
     * @brief Get raw red channel value (no clamping - for integral image calculations)
     * @return Raw red channel value
     */
    __host__ __device__ int getRRaw() const { return R; }

    /**
     * @brief Get raw green channel value (no clamping - for integral image calculations)
     * @return Raw green channel value
     */
    __host__ __device__ int getGRaw() const { return G; }

    /**
     * @brief Get raw blue channel value (no clamping - for integral image calculations)
     * @return Raw blue channel value
     */
    __host__ __device__ int getBRaw() const { return B; }
};

/**
 * @brief Scalar multiplication operator (scalar first)
 * @param t Scalar multiplier
 * @param p RGB color
 * @return Result of scalar multiplication
 */
__host__ __device__ inline RGB operator*(double t, RGB p) {
    return p * t;
}

/**
 * @brief Component-wise multiplication of two RGB colors
 * @param p First RGB color
 * @param t Second RGB color
 * @return Result of component-wise multiplication
 */
__host__ __device__ inline RGB operator*(const RGB &p, const RGB &t) {
    return RGB{
        p.getG() * t.getG(),
        p.getG() * t.getG(),
        p.getB() * t.getB(),
    };
}


} // namespace pixel