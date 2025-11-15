#pragma once

/*
  Wrappers for different image formats. This namespace may be appended in the
  future. For now works only with PPM-oriented images.

  All of the functions (public or private) in all classes should be __host__
  oriented. Every class should include and initialize `device_view::View` and
  have `get_device_view` function, which would return pointer to the
  `device_view::View` in order to pass this into kernel and work with
  device-oriented data.

*/

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

namespace render {

using namespace device_view;
namespace fs = std::filesystem;

// Abstract image class that should be a parent for all classes
class ImageBase {
    friend class cam::Camera;

protected:
    int height;
    int width;
    std::vector<pixel::RGB> host_content;
    pixel::RGB *device_content;

protected:
    // protected function-helpers for internal use-only
    // Copy image data from/to CPU/GPU buffers
    void copyToGPU();
    void copyFromGPU();
    void freeGPUMemory();

public:
    // Constructors (no need to init them) and destructor
    ImageBase() = default;
    virtual ~ImageBase() = 0;

public:
    // File-related functions (r, w, etc.)
    virtual void save(fs::path path) = 0;

    // Getters
    std::pair<int, int> get_size() const;
    int get_height();
    int get_width();
    dim3 get_grid_size(dim3 blockSize) const;
    pixel::RGB get_pixel(int x, int y);

    // Special getter that binds GPU memory and device_view
    View get_device_view() const { return View{device_content, width, height}; }
};

// PPM image format
class ImagePPM : public ImageBase {
public:
    // No any special fields for now
    ImagePPM(fs::path imagePath);
    void save(fs::path path);
    void boxBlur(int boxSize);
    void gaussianBlur(int boxSize);
    void gaussianBlurUnoptimized(int boxSize);
    void sobelEdgeDetection();
};

} // namespace render