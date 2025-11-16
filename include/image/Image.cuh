/**
 * @file Image.cuh
 * @brief Image processing classes for CUDA-based operations
 * 
 * Wrappers for different image formats. This namespace may be appended in the
 * future. For now works only with PPM-oriented images.
 *
 * All of the functions (public or private) in all classes should be __host__
 * oriented. Every class should include and initialize `device_view::View` and
 * have `get_device_view` function, which would return pointer to the
 * `device_view::View` in order to pass this into kernel and work with
 * device-oriented data.
 */

#pragma once

// Local includes
#include <filesystem>
#include <memory>
#include <vector>

// Local dependencies
#include "Pixel.cuh"
#include "View.cuh"

// Namespace for different images
namespace cam {
class Camera;
}

/**
 * @namespace render
 * @brief Namespace containing image rendering and processing classes
 */
namespace render {

using namespace device_view;
namespace fs = std::filesystem;

/**
 * @class ImageBase
 * @brief Abstract base class for all image types
 * 
 * This is the parent class for all image implementations. It provides
 * common functionality for memory management between CPU and GPU,
 * pixel access, and dimension queries.
 */
class ImageBase {
    friend class cam::Camera;

protected:
    int height;                          ///< Image height in pixels
    int width;                           ///< Image width in pixels
    std::vector<pixel::RGB> host_content; ///< CPU-side image data
    pixel::RGB *device_content;          ///< GPU-side image data pointer

protected:
    /**
     * @brief Copy image data from CPU to GPU
     */
    void copyToGPU();
    
    /**
     * @brief Copy image data from GPU to CPU
     */
    void copyFromGPU();
    
    /**
     * @brief Free GPU memory
     */
    void freeGPUMemory();

public:
    /**
     * @brief Default constructor
     */
    ImageBase() = default;
    
    /**
     * @brief Virtual destructor
     */
    virtual ~ImageBase() = 0;

public:
    /**
     * @brief Save image to file
     * @param path Filesystem path where to save the image
     */
    virtual void save(fs::path path) = 0;

    /**
     * @brief Get image dimensions
     * @return Pair of (width, height)
     */
    std::pair<int, int> get_size() const;
    
    /**
     * @brief Get image height
     * @return Image height in pixels
     */
    int get_height();
    
    /**
     * @brief Get image width
     * @return Image width in pixels
     */
    int get_width();
    
    /**
     * @brief Calculate grid size for CUDA kernel launch
     * @param blockSize CUDA block dimensions
     * @return Grid dimensions for kernel launch
     */
    dim3 get_grid_size(dim3 blockSize) const;
    
    /**
     * @brief Get pixel at specified coordinates
     * @param x X coordinate
     * @param y Y coordinate
     * @return RGB pixel value
     */
    pixel::RGB get_pixel(int x, int y);

    /**
     * @brief Get device view for GPU operations
     * @return View object containing GPU memory pointer and dimensions
     */
    View get_device_view() const { return View{device_content, width, height}; }
};

/**
 * @class ImagePPM
 * @brief PPM image format implementation
 * 
 * Concrete implementation of ImageBase for handling PPM image files.
 * Supports various image processing operations including blurring and
 * edge detection.
 */
class ImagePPM : public ImageBase {
public:
    /**
     * @brief Constructor that loads image from file
     * @param imagePath Path to PPM image file
     */
    ImagePPM(fs::path imagePath);
    
    /**
     * @brief Save image to PPM file
     * @param path Filesystem path where to save the image
     */
    void save(fs::path path);
    
    /**
     * @brief Apply box blur filter
     * @param boxSize Size of the blur box kernel
     */
    void boxBlur(int boxSize);
    
    /**
     * @brief Apply Gaussian blur filter (optimized)
     * @param boxSize Size of the Gaussian kernel
     */
    void gaussianBlur(int boxSize);
    
    /**
     * @brief Apply Gaussian blur filter (unoptimized version)
     * @param boxSize Size of the Gaussian kernel
     */
    void gaussianBlurUnoptimized(int boxSize);
    
    /**
     * @brief Apply Sobel edge detection filter
     */
    void sobelEdgeDetection();
};

} // namespace render