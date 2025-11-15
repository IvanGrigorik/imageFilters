// Cuda includes
#include "Pixel.cuh"
#include "image/Image.cuh"
#include "image/View.cuh"

// Cuda includes
#include <curand.h>
#include <curand_kernel.h>

__global__ void box_blur_kernel(device_view::View image, device_view::View output, int boxSize) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= image.get_width() || y >= image.get_height()) {
        return;
    }

    // Calculate the box blur by averaging pixels in the box
    int halfBox = boxSize / 2;
    int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    // Iterate over the box centered at (x, y)
    for (int dy = -halfBox; dy <= halfBox; dy++) {
        for (int dx = -halfBox; dx <= halfBox; dx++) {
            pixel::RGB pixel = image.get_pixel_safe(x + dx, y + dy);
            sumR += pixel.getR();
            sumG += pixel.getG();
            sumB += pixel.getB();
            count++;
        }
    }

    // Calculate average and write to output
    pixel::RGB blurred(sumR / count, sumG / count, sumB / count);
    output.set_pixel(x, y, blurred);
}

// Compute row-wise prefix sum (inclusive scan)
__global__ void compute_row_prefix_sum(device_view::View input, device_view::View output) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int width = input.get_width();
    int height = input.get_height();

    if (y >= height)
        return;

    int sumR = 0, sumG = 0, sumB = 0;
    for (int x = 0; x < width; x++) {
        pixel::RGB p = input.get_pixel(x, y);
        sumR += p.getR();
        sumG += p.getG();
        sumB += p.getB();
        output.set_pixel(x, y, pixel::RGB(sumR, sumG, sumB));
    }
}

// Compute column-wise prefix sum (inclusive scan) on already row-summed data
__global__ void compute_col_prefix_sum(device_view::View input, device_view::View output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int width = input.get_width();
    int height = input.get_height();

    if (x >= width)
        return;

    int sumR = 0, sumG = 0, sumB = 0;
    for (int y = 0; y < height; y++) {
        pixel::RGB p = input.get_pixel(x, y);
        // Use raw values since input already contains cumulative sums from row prefix
        sumR += p.getRRaw();
        sumG += p.getGRaw();
        sumB += p.getBRaw();
        output.set_pixel(x, y, pixel::RGB(sumR, sumG, sumB));
    }
}

// Use integral image to compute box blur in O(1) per pixel
__global__ void integral_box_blur_kernel(device_view::View integral, device_view::View output, int boxSize) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int width = integral.get_width();
    int height = integral.get_height();

    if (x >= width || y >= height)
        return;

    int halfBox = boxSize / 2;

    // Calculate box boundaries (inclusive)
    int x_min = max(0, x - halfBox);
    int y_min = max(0, y - halfBox);
    int x_max = min(width - 1, x + halfBox);
    int y_max = min(height - 1, y + halfBox);

    // For inclusive prefix sum, to get sum in rectangle [x_min, x_max] x [y_min, y_max]:
    // sum = integral[x_max][y_max] - integral[x_min-1][y_max] - integral[x_max][y_min-1] + integral[x_min-1][y_min-1]

    pixel::RGB sum_D = integral.get_pixel(x_max, y_max);

    pixel::RGB sum_B_corner = (y_min > 0) ? integral.get_pixel(x_max, y_min - 1) : pixel::RGB(0, 0, 0);
    pixel::RGB sum_C_corner = (x_min > 0) ? integral.get_pixel(x_min - 1, y_max) : pixel::RGB(0, 0, 0);
    pixel::RGB sum_A_corner = (x_min > 0 && y_min > 0) ? integral.get_pixel(x_min - 1, y_min - 1) : pixel::RGB(0, 0, 0);

    // Box sum = D - B - C + A (use raw values to avoid modulo)
    int sumR = sum_D.getRRaw() - sum_B_corner.getRRaw() - sum_C_corner.getRRaw() + sum_A_corner.getRRaw();
    int sumG = sum_D.getGRaw() - sum_B_corner.getGRaw() - sum_C_corner.getGRaw() + sum_A_corner.getGRaw();
    int sumBlue = sum_D.getBRaw() - sum_B_corner.getBRaw() - sum_C_corner.getBRaw() + sum_A_corner.getBRaw();

    // Calculate actual box area
    int area = (x_max - x_min + 1) * (y_max - y_min + 1);

    if (area > 0) {
        output.set_pixel(x, y, pixel::RGB(sumR / area, sumG / area, sumBlue / area));
    }
}

// Gaussian blur - horizontal pass
__global__ void gaussian_blur_horizontal(device_view::View input, device_view::View output, float *kernel,
                                         int kernelRadius) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int width = input.get_width();
    int height = input.get_height();

    if (x >= width || y >= height)
        return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    // Apply horizontal Gaussian kernel
    for (int i = -kernelRadius; i <= kernelRadius; i++) {
        int sampleX = x + i;
        // Clamp to image boundaries
        if (sampleX < 0)
            sampleX = 0;
        if (sampleX >= width)
            sampleX = width - 1;

        pixel::RGB pixel = input.get_pixel(sampleX, y);
        float weight = kernel[i + kernelRadius];

        sumR += pixel.getR() * weight;
        sumG += pixel.getG() * weight;
        sumB += pixel.getB() * weight;
    }

    output.set_pixel(
        x, y, pixel::RGB(static_cast<int>(sumR + 0.5f), static_cast<int>(sumG + 0.5f), static_cast<int>(sumB + 0.5f)));
}

// Gaussian blur - vertical pass
__global__ void gaussian_blur_vertical(device_view::View input, device_view::View output, float *kernel,
                                       int kernelRadius) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int width = input.get_width();
    int height = input.get_height();

    if (x >= width || y >= height)
        return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    // Apply vertical Gaussian kernel
    for (int i = -kernelRadius; i <= kernelRadius; i++) {
        int sampleY = y + i;
        // Clamp to image boundaries
        if (sampleY < 0)
            sampleY = 0;
        if (sampleY >= height)
            sampleY = height - 1;

        pixel::RGB pixel = input.get_pixel(x, sampleY);
        float weight = kernel[i + kernelRadius];

        sumR += pixel.getR() * weight;
        sumG += pixel.getG() * weight;
        sumB += pixel.getB() * weight;
    }

    output.set_pixel(
        x, y, pixel::RGB(static_cast<int>(sumR + 0.5f), static_cast<int>(sumG + 0.5f), static_cast<int>(sumB + 0.5f)));
}

// Unoptimized 2D Gaussian blur - applies full 2D kernel (no separability optimization)
__global__ void gaussian_blur_2d_kernel(device_view::View input, device_view::View output, float *kernel,
                                        int kernelRadius) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int width = input.get_width();
    int height = input.get_height();

    if (x >= width || y >= height)
        return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    // Apply 2D Gaussian kernel by iterating over the entire kernel window
    for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
        for (int dx = -kernelRadius; dx <= kernelRadius; dx++) {
            int sampleX = x + dx;
            int sampleY = y + dy;

            // Clamp to image boundaries
            if (sampleX < 0)
                sampleX = 0;
            if (sampleX >= width)
                sampleX = width - 1;
            if (sampleY < 0)
                sampleY = 0;
            if (sampleY >= height)
                sampleY = height - 1;

            pixel::RGB pixel = input.get_pixel(sampleX, sampleY);

            // Calculate 2D Gaussian weight as product of 1D weights
            // kernel is stored as 1D array, so we compute 2D weight from 1D values
            float weightX = kernel[dx + kernelRadius];
            float weightY = kernel[dy + kernelRadius];
            float weight2D = weightX * weightY;

            sumR += pixel.getR() * weight2D;
            sumG += pixel.getG() * weight2D;
            sumB += pixel.getB() * weight2D;
        }
    }

    output.set_pixel(
        x, y, pixel::RGB(static_cast<int>(sumR + 0.5f), static_cast<int>(sumG + 0.5f), static_cast<int>(sumB + 0.5f)));
}

// Sobel edge detection kernel
__global__ void sobel_edge_detection_kernel(device_view::View input, device_view::View output) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int width = input.get_width();
    int height = input.get_height();

    if (x >= width || y >= height)
        return;

    // Sobel operators (kernels)
    // Gx (horizontal edge detection)
    const int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    // Gy (vertical edge detection)
    const int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    float gradientX_R = 0.0f, gradientX_G = 0.0f, gradientX_B = 0.0f;
    float gradientY_R = 0.0f, gradientY_G = 0.0f, gradientY_B = 0.0f;

    // Apply Sobel operators
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sampleX = x + dx;
            int sampleY = y + dy;

            // Clamp to image boundaries
            if (sampleX < 0) sampleX = 0;
            if (sampleX >= width) sampleX = width - 1;
            if (sampleY < 0) sampleY = 0;
            if (sampleY >= height) sampleY = height - 1;

            pixel::RGB pixel = input.get_pixel(sampleX, sampleY);
            
            int gx_weight = Gx[dy + 1][dx + 1];
            int gy_weight = Gy[dy + 1][dx + 1];

            // Apply weights for X gradient
            gradientX_R += pixel.getR() * gx_weight;
            gradientX_G += pixel.getG() * gx_weight;
            gradientX_B += pixel.getB() * gx_weight;

            // Apply weights for Y gradient
            gradientY_R += pixel.getR() * gy_weight;
            gradientY_G += pixel.getG() * gy_weight;
            gradientY_B += pixel.getB() * gy_weight;
        }
    }

    // Calculate gradient magnitude using Euclidean distance
    float magnitudeR = sqrtf(gradientX_R * gradientX_R + gradientY_R * gradientY_R);
    float magnitudeG = sqrtf(gradientX_G * gradientX_G + gradientY_G * gradientY_G);
    float magnitudeB = sqrtf(gradientX_B * gradientX_B + gradientY_B * gradientY_B);

    // Clamp values to 0-255 range
    int outR = min(255, max(0, static_cast<int>(magnitudeR)));
    int outG = min(255, max(0, static_cast<int>(magnitudeG)));
    int outB = min(255, max(0, static_cast<int>(magnitudeB)));

    output.set_pixel(x, y, pixel::RGB(outR, outG, outB));
}
