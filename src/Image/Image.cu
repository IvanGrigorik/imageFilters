// Local includes
#include "helper.cuh"
#include "image/Image.cuh"
#include "image/ImageCuda.cuh"

// Standard includes
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>

// CUDA includes

namespace render {
using namespace std;
namespace fs = std::filesystem;

//// `ImageBase` class generic functions
/*** ImagePPM class functions ***/
ImageBase::~ImageBase() {
    freeGPUMemory();
};

/* Setters and getters */
// returns: pair image `width` and `heigh` in the ```std::pair<int, int>```
std::pair<int, int> ImageBase::get_size() const {
    return {this->width, this->height};
}

// Return grid size with assumption that block size is 16x16
dim3 ImageBase::get_grid_size(dim3 block_size) const {
    return dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
}

pixel::RGB ImageBase::get_pixel(int x, int y) {
    return host_content[y * this->width + x];
}

/* Getters and setters */
int ImageBase::get_height() {
    return this->height;
}

int ImageBase::get_width() {
    return this->width;
}

/* Mics */
// ImageBase file writer
void ImagePPM::save(fs::path path) {
    if (!fs::exists(path.parent_path())) {
        fs::create_directories(path.parent_path());
    }
    // synchronize device and host vectors
    ofstream out_file(path);

    // ImageBase preamble
    out_file << "P3\n" << this->width << ' ' << this->height << "\n255\n";

    // Load main content to the buffer (TODO: change to array-to-file mapping later)
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            const pixel::RGB pixel = this->get_pixel(j, i);
            out_file << pixel.getR() << ' ' << pixel.getG() << ' ' << pixel.getB() << '\n';
        }
    }
    out_file.close();
}

/* Private functions for internal-use only */
// Copy image matrix from CPU to GPU
void ImageBase::copyToGPU() {
    if (host_content.empty() || height == 0 || width == 0) {
        throw std::runtime_error("No image data to copy to GPU");
    }

    freeGPUMemory();

    size_t data_size = height * width * sizeof(pixel::RGB);

    CUDA_CHECK(cudaMalloc((pixel::RGB **)&device_content, data_size));

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, device_content));
    std::cout << "Device memory type: " << attr.type << std::endl;

    CUDA_CHECK(cudaMemcpy(device_content, host_content.data(), data_size, cudaMemcpyHostToDevice));
}

// Copy image matrix from GPU to CPU
void ImageBase::copyFromGPU() {
    if (!device_content || height == 0 || width == 0) {
        throw std::runtime_error("No GPU data to copy from");
    }

    size_t data_size = height * width * sizeof(pixel::RGB);

    // Ensure CPU buffer is properly sized
    if (host_content.size() != height * width) {
        host_content.resize(height * width);
    }

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, device_content));
    if (attr.type != 2) {
        printf("CUDA memory is invalid at %s:%d; \n\tMemory type: %d\n", __FILE__, __LINE__, attr.type);
        exit(-1);
    }
    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(host_content.data(), device_content, data_size, cudaMemcpyDeviceToHost));
}

// Free GPU image buffer
void ImageBase::freeGPUMemory() {
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, device_content));

    if (device_content && attr.type == 2) {
        CUDA_CHECK(cudaFree(device_content));
        device_content = nullptr;
    }
}

/*** ImagePPM class functions ***/
ImagePPM::ImagePPM(fs::path imagePath) {
    // load PPM image structure
    std::ifstream imageFile(imagePath);

    if (!imageFile.is_open()) {
        std::cerr << "Failed to open file: " << imagePath << "\n";
        return;
    }
    std::string magic;
    imageFile >> magic;
    if (magic != "P3") {
        std::cerr << "Unsupported PPM format: " << magic << " (expected P3)\n";
        return;
    }

    // Skip comments if any (# ...)
    imageFile >> std::ws;
    while (imageFile.peek() == '#') {
        std::string comment;
        std::getline(imageFile, comment);
        imageFile >> std::ws;
    }

    int max_px;
    imageFile >> width >> height >> max_px;

    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid image dimensions\n";
        return;
    }
    host_content.reserve(height * width);
    int r, g, b;
    while (imageFile >> r >> g >> b) {
        host_content.emplace_back(r, g, b);
    }
    imageFile.close();
    size_t data_size = height * width * sizeof(pixel::RGB);
    CUDA_CHECK(cudaMalloc((pixel::RGB **)&device_content, data_size));
    CUDA_CHECK(cudaMemcpy(device_content, host_content.data(), host_content.size() * sizeof(pixel::RGB),
                          cudaMemcpyHostToDevice));
}

/* Naive box blur. Each kernel computes its value based on window */
void ImagePPM::boxBlur(int boxSize) {
    if (!height || !width) {
        throw std::runtime_error("Cannot blur image with zero dimensions");
    }
    if (boxSize < 1) {
        throw std::runtime_error("Box size must be at least 1");
    }

    copyToGPU();

    // Allocate output buffer on GPU
    pixel::RGB *device_output;
    size_t data_size = height * width * sizeof(pixel::RGB);
    CUDA_CHECK(cudaMalloc((pixel::RGB **)&device_output, data_size));

    dim3 block_size(16, 16);
    dim3 grid_size = get_grid_size(block_size);

    // Create views for input and output
    View input_view = get_device_view();
    View output_view(device_output, width, height);

    // For small box sizes, use naive approach
    if (boxSize < 10) {
        box_blur_kernel<<<grid_size, block_size>>>(input_view, output_view, boxSize);
    } else {
        // For large box sizes, use integral image approach
        pixel::RGB *device_integral;
        pixel::RGB *device_temp;
        CUDA_CHECK(cudaMalloc((pixel::RGB **)&device_integral, data_size));
        CUDA_CHECK(cudaMalloc((pixel::RGB **)&device_temp, data_size));
        
        View integral_view(device_integral, width, height);
        View temp_view(device_temp, width, height);
        
        // Step 1: Compute row-wise prefix sum
        dim3 block_1d(256);
        dim3 grid_rows((height + block_1d.x - 1) / block_1d.x);
        compute_row_prefix_sum<<<grid_rows, block_1d>>>(input_view, temp_view);
        CUDA_CHECK(cudaGetLastError());
        
        // Step 2: Compute column-wise prefix sum on row-summed data
        dim3 grid_cols((width + block_1d.x - 1) / block_1d.x);
        compute_col_prefix_sum<<<grid_cols, block_1d>>>(temp_view, integral_view);
        CUDA_CHECK(cudaGetLastError());
        
        // Step 3: Use integral image to compute box blur
        integral_box_blur_kernel<<<grid_size, block_size>>>(integral_view, output_view, boxSize);
        CUDA_CHECK(cudaGetLastError());
        
        // Free temporary buffers
        CUDA_CHECK(cudaFree(device_integral));
        CUDA_CHECK(cudaFree(device_temp));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to device_content
    CUDA_CHECK(cudaMemcpy(device_content, device_output, data_size, cudaMemcpyDeviceToDevice));

    // Free temporary output buffer
    CUDA_CHECK(cudaFree(device_output));

    copyFromGPU();
}

void ImagePPM::gaussianBlur(int boxSize) {
    if (!height || !width) {
        throw std::runtime_error("Cannot blur image with zero dimensions");
    }
    if (boxSize < 1) {
        throw std::runtime_error("Box size must be at least 1");
    }

    copyToGPU();

    // Calculate sigma based on boxSize (rule of thumb: radius ≈ 3*sigma)
    float sigma = boxSize / 6.0f;
    if (sigma < 0.5f) sigma = 0.5f;
    
    // Calculate kernel radius (typically 3*sigma gives good approximation)
    int kernelRadius = static_cast<int>(std::ceil(3.0f * sigma));
    int kernelSize = 2 * kernelRadius + 1;
    
    // Generate 1D Gaussian kernel on CPU
    std::vector<float> host_kernel(kernelSize);
    float sum = 0.0f;
    
    for (int i = 0; i < kernelSize; i++) {
        int x = i - kernelRadius;
        float value = std::exp(-(x * x) / (2.0f * sigma * sigma));
        host_kernel[i] = value;
        sum += value;
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        host_kernel[i] /= sum;
    }
    
    // Copy kernel to GPU
    float *device_kernel;
    CUDA_CHECK(cudaMalloc(&device_kernel, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_kernel, host_kernel.data(), kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate temporary and output buffers on GPU
    pixel::RGB *device_temp;
    pixel::RGB *device_output;
    size_t data_size = height * width * sizeof(pixel::RGB);
    CUDA_CHECK(cudaMalloc(&device_temp, data_size));
    CUDA_CHECK(cudaMalloc(&device_output, data_size));
    
    dim3 block_size(16, 16);
    dim3 grid_size = get_grid_size(block_size);
    
    // Create views
    View input_view = get_device_view();
    View temp_view(device_temp, width, height);
    View output_view(device_output, width, height);
    
    // Apply separable Gaussian blur
    // Step 1: Horizontal pass
    gaussian_blur_horizontal<<<grid_size, block_size>>>(input_view, temp_view, device_kernel, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Vertical pass
    gaussian_blur_vertical<<<grid_size, block_size>>>(temp_view, output_view, device_kernel, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to device_content
    CUDA_CHECK(cudaMemcpy(device_content, device_output, data_size, cudaMemcpyDeviceToDevice));
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(device_kernel));
    CUDA_CHECK(cudaFree(device_temp));
    CUDA_CHECK(cudaFree(device_output));
    
    copyFromGPU();
}

void ImagePPM::gaussianBlurUnoptimized(int boxSize) {
    if (!height || !width) {
        throw std::runtime_error("Cannot blur image with zero dimensions");
    }
    if (boxSize < 1) {
        throw std::runtime_error("Box size must be at least 1");
    }

    copyToGPU();

    // Calculate sigma based on boxSize (rule of thumb: radius ≈ 3*sigma)
    float sigma = boxSize / 6.0f;
    if (sigma < 0.5f) sigma = 0.5f;
    
    // Calculate kernel radius (typically 3*sigma gives good approximation)
    int kernelRadius = static_cast<int>(std::ceil(3.0f * sigma));
    int kernelSize = 2 * kernelRadius + 1;
    
    // Generate 1D Gaussian kernel on CPU
    std::vector<float> host_kernel(kernelSize);
    float sum = 0.0f;
    
    for (int i = 0; i < kernelSize; i++) {
        int x = i - kernelRadius;
        float value = std::exp(-(x * x) / (2.0f * sigma * sigma));
        host_kernel[i] = value;
        sum += value;
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        host_kernel[i] /= sum;
    }
    
    // Copy kernel to GPU
    float *device_kernel;
    CUDA_CHECK(cudaMalloc(&device_kernel, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_kernel, host_kernel.data(), kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate output buffer on GPU
    pixel::RGB *device_output;
    size_t data_size = height * width * sizeof(pixel::RGB);
    CUDA_CHECK(cudaMalloc(&device_output, data_size));
    
    dim3 block_size(16, 16);
    dim3 grid_size = get_grid_size(block_size);
    
    // Create views
    View input_view = get_device_view();
    View output_view(device_output, width, height);
    
    // Apply 2D Gaussian blur directly (no separability)
    gaussian_blur_2d_kernel<<<grid_size, block_size>>>(input_view, output_view, device_kernel, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to device_content
    CUDA_CHECK(cudaMemcpy(device_content, device_output, data_size, cudaMemcpyDeviceToDevice));
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(device_kernel));
    CUDA_CHECK(cudaFree(device_output));
    
    copyFromGPU();
}

} // namespace render
