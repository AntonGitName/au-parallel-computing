//
// Created by antonpp on 16.01.17.
//

#ifndef AU_PARALLEL_COMPUTING_IMAGEPROCESSOR_H
#define AU_PARALLEL_COMPUTING_IMAGEPROCESSOR_H

#include <tbb/flow_graph.h>

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ostream>
#include <iostream>

#include "Image.h"

class ImageProcessor {
public:
    ImageProcessor(const std::vector<Image>& images, char pixel_value, size_t image_parallel,
                   std::string log_fname);
    void process();
private:
    std::vector<Image> images;
    size_t added_images = 0;
    tbb::flow::graph flow_graph;
    std::ofstream average_pixel_log;
    std::shared_ptr<tbb::flow::source_node<Image>> source_ptr;
};


#endif //AU_PARALLEL_COMPUTING_IMAGEPROCESSOR_H
