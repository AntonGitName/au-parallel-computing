#include <iostream>
#include <vector>

#include "ImageProcessor.h"

namespace {
    void usage(char const *name) {
        std::cout << "Usage: " << name << " OPTIONS" << std::endl;
        std::cout << std::endl;
        std::cout << "OPTIONS:" << std::endl;
        std::cout << "  -b NUM      brightness value to search for [0..255] " << std::endl;
        std::cout << "  -l LIMIT    max number of proccessing images at a time" << std::endl;
        std::cout << "  -f FILE     log file" << std::endl;
    }

    std::vector<Image> create_images(size_t n) {
        std::vector<Image> images;
        for (size_t i = 0; i < n; ++i) {
            images.push_back(Image(512, 512));
        }
        return images;
    }
}

int main(int argc, char **argv) {
    int pixel_to_search = 128;
    size_t parallel_images = 4;
    std::string log_fname = "flow-graph.log";

    for (int i = 1; i < argc; i += 2) {
        std::string flag = argv[i];
        if (flag == "-h" || flag == "--help") {
            usage(argv[0]);
            return 0;
        } else if (flag == "-f") {
            log_fname = argv[i + 1];
        } else if (flag == "-b") {
            pixel_to_search = std::stoi(argv[i + 1]);
        } else if (flag == "-l") {
            parallel_images = std::stoul(argv[i + 1]);
        } else {
            usage(argv[0]);
            exit(1);
        }
    }

    Image::pixel_t brightness = (Image::pixel_t) pixel_to_search;

    ImageProcessor ip(create_images(64), brightness, parallel_images, log_fname);
    ip.process();

    return 0;
}