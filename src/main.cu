#include <cstdlib>
#include <cuda.h>
#include <iostream>

// Local-based includes
#include "image/Image.cuh"
namespace fs = std::filesystem;
using namespace std;

void parse_args(int argc, char *argv[], fs::path &path) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return;
    }
    fs::path input_path = argv[1];
    fs::path abs_path = fs::weakly_canonical(input_path);
    if (!fs::exists(abs_path)) {
        std::cerr << "Error: file not found\n";
    }
    path = abs_path;
}

int main(int argc, char *argv[]) {
    fs::path imagePath;
    parse_args(argc, argv, imagePath);
    cout << imagePath << '\n';

    render::ImagePPM image;

    // // TODO: remove later. temporary solution to debug faster
    fs::path core_dir = fs::path(__FILE__).parent_path().parent_path();
    fs::path out = core_dir / "out/image.ppm";
    fs::path converted = core_dir / "out/converted.png";
    std::string cmd = "convert " + out.string() + " " + converted.string();
    system(cmd.c_str());
}