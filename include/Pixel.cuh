// Wrapper class for pixels of different colors formats

#pragma once
#include <stdio.h>

// Declare the image types in here (do not need to know anything except
// declaration)
namespace render {
class ImageBase;
}

// Namespace for different color formats
namespace pixel {

// Classic RGB format
class RGB {
private:
    int R;
    int G;
    int B;

public:
    __host__ __device__ RGB() {};
    __host__ __device__ RGB(int r, int g, int b) : R(r), G(g), B(b) {};

    __host__ __device__ RGB(double r, double g, double b)
        : R(int(r * 255.999)), G(int(g * 255.999)), B(int(b * 255.999)) {};

    __host__ __device__ RGB(float r, float g, float b)
        : R(int(r * 255.9999)), G(int(g * 255.9999)), B(int(b * 255.9999)) {};

    // Befriend all of the pixel classes
    friend class render::ImageBase;

    // Operators
    __host__ __device__ RGB operator+(const RGB &c) {
        R += c.R;
        G += c.G;
        B += c.B;
        return *this;
    }

    __host__ __device__ RGB operator*(double t) {
        R *= t;
        G *= t;
        B *= t;
        return *this;
    }

    __host__ __device__ RGB operator%(int t) {
        R = R % t;
        G = G % t;
        B = B % t;
        return *this;
    }

    __host__ __device__ RGB get() const { return *this; }

    // getters
    __host__ __device__ int getR() const { return R % 256; }

    __host__ __device__ int getG() const { return G % 256; }

    __host__ __device__ int getB() const { return B % 256; }
};

__host__ __device__ inline RGB operator*(double t, RGB p) {
    return p * t;
}

__host__ __device__ inline RGB operator*(const RGB &p, const RGB &t) {
    return RGB{
        p.getG() * t.getG(),
        p.getG() * t.getG(),
        p.getB() * t.getB(),
    };
}


} // namespace pixel