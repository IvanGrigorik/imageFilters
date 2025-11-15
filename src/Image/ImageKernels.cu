// Cuda includes
#include "Pixel.cuh"
#include "image/View.cuh"
#include "image/Image.cuh"

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
    
    if (y >= height) return;
    
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
    
    if (x >= width) return;
    
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
    
    if (x >= width || y >= height) return;
    
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

// __global__ void fill_image_plain(device_view::View image, pixel::RGB color) {
//     int x = blockDim.x * blockIdx.x + threadIdx.x;
//     int y = blockDim.y * blockIdx.y + threadIdx.y;

//     if (x >= image.get_width() || y >= image.get_height()) {
//         return;
//     }
    // image.set_pixel(x, y, color);
// }

// __global__ void fill_image_random(device_view::View image, unsigned long long seed) {
//     int x = blockDim.x * blockIdx.x + threadIdx.x;
//     int y = blockDim.y * blockIdx.y + threadIdx.y;

//     if (x >= image.get_width() || y >= image.get_height()) {
//         return;
//     }
//     int idx = y * image.get_width() + x;
//     curandState localState;
//     curand_init(seed, idx, 0, &localState);

//     pixel::RGB p(curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState));
//     image.set_pixel(x, y, p);
// }

// __global__ void fill_image_gradient(device_view::View image) {
//     int x = blockDim.x * blockIdx.x + threadIdx.x;
//     int y = blockDim.y * blockIdx.y + threadIdx.y;

//     if (x >= image.get_width() || y >= image.get_height()) {
//         return;
//     }
//     auto r = double(x) / (image.get_width() - 1);
//     auto g = double(y) / (image.get_height() - 1);
//     auto b = 0.0;
//     image.set_pixel(x, y, {r, g, b});
// }