// Local includes
#include "helper.cuh"
#include "image/Image.cuh"
#include "image/ImageCuda.cuh"

// Standard includes
#include <algorithm>
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
    std::ifstream imageFile(imagePath, std::ios::binary);

    if (!imageFile.is_open()) {
        std::cerr << "Failed to open file: " << imagePath << "\n";
        return;
    }
    std::string magic;
    imageFile >> magic;
    if (magic != "P6") {
        std::cerr << "Unsupported PPM format: " << magic << "\n";
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
    // skip the newline after header
    imageFile.ignore(1);
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

void ImagePPM::boxBlur(int boxSize) {
    if (!height || !width) {
        throw std::runtime_error("Cannot fill image with zero dimensions");
    }

    copyToGPU();
    dim3 block_size(16, 16);
    dim3 grid_size = get_grid_size(block_size);

    float weight = 1.0 / float(boxSize * boxSize);
    // Device integral image for fast box sum
    float *d_integral;

    


    // Hi Blur!
    box_blur_kernel<<<grid_size, block_size>>>(this->get_device_view());

    cudaDeviceSynchronize();

    copyFromGPU();
}

} // namespace render
