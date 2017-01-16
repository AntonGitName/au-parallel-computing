//
// Created by antonpp on 16.01.17.
//

#ifndef AU_PARALLEL_COMPUTING_IMAGE_H
#define AU_PARALLEL_COMPUTING_IMAGE_H

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <limits>

class Image {
public:
    typedef unsigned char pixel_t;
    typedef std::pair<int, int> pos_t;

    Image(size_t w = 0, size_t h = 0);
    std::vector<pos_t> get_border(size_t pixel_index) const;
    std::vector<pixel_t> get_pixels() const;
    pixel_t get_pixel(pos_t pos) const;
    pixel_t get_pixel(size_t pos) const;
    size_t get_width() const;
    size_t get_height() const;

    static pixel_t invert_pixel(pixel_t c);
private:
    size_t width;
    size_t height;
    std::vector<pixel_t> pixels;

};


#endif //AU_PARALLEL_COMPUTING_IMAGE_H