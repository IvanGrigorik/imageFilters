#include <cstdlib>
#include <cuda.h>
#include <iostream>

// Local-based includes
#include "image/Image.cuh"
namespace fs = std::filesystem;
using namespace std;

// parses main function arguments
pair<fs::path, int> parse_args(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <blur_size>\n";
        exit(1);
    }
    fs::path input_path = argv[1];
    fs::path abs_path = fs::weakly_canonical(input_path);
    if (!fs::exists(abs_path)) {
        std::cerr << "Error: file not found\n";
    }
    int boxSize = stoi(argv[2]);
    return {abs_path, boxSize};
}

int main(int argc, char *argv[]) {
    auto [imagePath, blurSize] = parse_args(argc, argv);
    cout << imagePath << '\n';

    // Apply Box blurring to the output image
    render::ImagePPM image(imagePath);
    image.boxBlur(blurSize);
    // image.gaussianBlur(blurSize);

    // Save the result
    fs::path core_dir = fs::path(__FILE__).parent_path().parent_path();
    fs::path out = core_dir / "out/image.ppm";
    image.save(out);
    cout << "Blurred image saved to " << out << '\n';
    
    // Convert to PNG for easier viewing
    fs::path converted = core_dir / "out/converted.png";
    std::string cmd = "convert " + out.string() + " " + converted.string();
    system(cmd.c_str());
}