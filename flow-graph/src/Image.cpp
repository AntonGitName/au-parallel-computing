//
// Created by antonpp on 16.01.17.
//

#include "Image.h"

Image::Image(size_t w, size_t h) {
    generate(w, h);
}

std::vector<Image::pos_t> Image::get_border(size_t pixel_index) const {
    int x = (int) (pixel_index / width);
    int y = (int) (pixel_index % width);
    std::vector<pos_t> border = {{x - 1, y - 1}, {x, y - 1}, {x + 1, y - 1},
                                 {x - 1, y},                 {x + 1, y},
                                 {x - 1, y + 1}, {x, y + 1}, {x + 1, y + 1}
    };
    std::vector<pos_t> filtered_border;
    auto is_valid_pos = [&](pos_t pos) {
        return pos.first >= 0 && pos.first < height && pos.second >= 0 && pos.second < width;
    };
    std::copy_if(border.begin(), border.end(), std::back_inserter(filtered_border), is_valid_pos);
    return filtered_border;
}

const std::vector<Image::pixel_t>& Image::get_pixels() const {
    return pixels;
}

Image::pixel_t Image::invert_pixel(pixel_t c) {
    return std::numeric_limits<pixel_t>::max() - c;
}

Image::pixel_t Image::get_pixel(pos_t pos) const {
    return pixels[pos.first * width + pos.second];
}

Image::pixel_t Image::get_pixel(size_t pos) const {
    return pixels[pos];
}

void Image::generate(size_t w, size_t h) {
    width = w;
    height = h;
    pixels.assign(w * h, 0);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = (pixel_t) rand();
    }
}
