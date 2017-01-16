#include <tbb/flow_graph.h>

#include <iostream>
#include <vector>

#include "ImageProcessor.h"

std::vector<Image> create_images(size_t n) {
    std::vector<Image> images;
    for (size_t i = 0; i < n; ++i) {
        images.push_back(Image(512, 512));
    }
    return images;
}

void usage(char const *name) {
    std::cerr << "Usage: " << name;
    std::cerr << " [-f filename] ";
    std::cerr << " [-b value] ";
    std::cerr << " [-l number] ";
    std::cerr << std::endl << std::endl;
    std::cerr << "OPTIONS\n";
    std::cerr << "\t-f filename\t Use it to specify a path where program log with average values will be written (default `flow-graph.log`)." << std::endl;
    std::cerr << "\t-b value\t Use it to set brightness value that will be searched in image (default `128`)." << std::endl;
    std::cerr << "\t-l number\t This option sets number of images processed simultaneously (default `4`)." << std::endl;
}

int main(int argc, char **argv) {
    int pixel_to_search = 128;
    size_t parallel_images = 4;
    std::string log_fname = "flow-graph.log";

    for (int i = 1; i < argc; i += 2) {
        std::string flag = argv[i];
        std::string value = argv[i + 1];
        if (flag == "-f") {
            log_fname = value;
        }
        if (flag == "-b") {
            pixel_to_search = std::stoi(value);
        }
        if (flag == "-l") {
            parallel_images = std::stoul(value);
        }

    }

    Image::pixel_t brightness = (Image::pixel_t) pixel_to_search;

    if (parallel_images < 0) {
        usage(argv[0]);
        return -1;
    }

    ImageProcessor ip(create_images(64), brightness, parallel_images, log_fname);
    ip.process();
    return 0;
}