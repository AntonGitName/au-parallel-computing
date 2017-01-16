//
// Created by antonpp on 16.01.17.
//

#include "ImageProcessor.h"

using namespace tbb::flow;

namespace {
    
    typedef std::pair<Image, std::vector<size_t> > f_node_result_t;
    typedef tuple<f_node_result_t, f_node_result_t, f_node_result_t> first_stage_tuple;
    
    std::vector<size_t> get_indices(const Image &image, char pixel_value) {
        std::vector<size_t> result;
        for (size_t i = 0; i < image.get_pixels().size(); ++i) {
            if (image.get_pixels()[i] == pixel_value) {
                result.push_back(i);
            }
        }

        return result;
    }
    
    void invert_border(const Image& image, size_t selected_pixel) {
        const auto &border = image.get_border(selected_pixel);
        for (auto index: border) {
            Image::invert_pixel(image.get_pixel(index));
            // some useful code here
        }
    }

    std::vector<Image::pixel_t> get_selected_pixels(f_node_result_t holder) {
        std::vector<Image::pixel_t> result;
        for (auto x : holder.second) {
            result.push_back(holder.first.get_pixel(x));
        }
        return result;
    }
}

void ImageProcessor::process() {
    source_ptr -> activate();
    flow_graph.wait_for_all();
}

ImageProcessor::ImageProcessor(const std::vector<Image> &images, char pixel_value, size_t image_parallel,
                               std::string log_fname) : images(images), average_pixel_log(log_fname) {
    auto source_f = [&](Image &image) {
//        std::cout << added_images;
//        std::cout.flush();
        image = this->images[0];
//        ++added_images;
        return true;//added_images < this->images.size();
    };

    auto max_pixel_f = [](const Image &image) {
        auto max = *std::max_element(image.get_pixels().begin(), image.get_pixels().end());
        return std::make_pair(image, get_indices(image, max));
    };
    auto min_pixel_f = [](const Image &image) {
        auto min = *std::min_element(image.get_pixels().begin(), image.get_pixels().end());
        return std::make_pair(image, get_indices(image, min));
    };

    auto search_pixel_f = [&pixel_value](const Image &image) {
        return std::make_pair(image, get_indices(image, pixel_value));
    };

    auto invert_selected_f = [](const first_stage_tuple &t) {
        auto image_with_selected_pixels = get<0>(t);
        for (auto index: image_with_selected_pixels.second) {
            invert_border(image_with_selected_pixels.first, index);
        }

        image_with_selected_pixels = get<1>(t);
        for (auto index: image_with_selected_pixels.second) {
            invert_border(image_with_selected_pixels.first, index);
        }

        image_with_selected_pixels = get<2>(t);
        for (auto index: image_with_selected_pixels.second) {
            invert_border(image_with_selected_pixels.first, index);
        }

        return true;
    };

    auto average_selected_f = [&](first_stage_tuple const &t) {

        std::vector<Image::pixel_t> selected_pixels;
        auto image_with_selected_pixels = get<0>(t);
        auto r = get_selected_pixels(image_with_selected_pixels);
        std::copy(r.begin(), r.end(), std::back_inserter(selected_pixels));

        image_with_selected_pixels = get<1>(t);
        r = get_selected_pixels(image_with_selected_pixels);
        std::copy(r.begin(), r.end(), std::back_inserter(selected_pixels));

        image_with_selected_pixels = get<2>(t);
        r = get_selected_pixels(image_with_selected_pixels);
        std::copy(r.begin(), r.end(), std::back_inserter(selected_pixels));

        size_t value = std::accumulate<std::vector<Image::pixel_t>::iterator, size_t>(selected_pixels.begin(), selected_pixels.end(), 0);
        Image::pixel_t average_value = (Image::pixel_t) (value / selected_pixels.size());

        average_pixel_log << average_value << std:: endl;

        return true;
    };

    auto stub_continue = [](tuple<bool, bool> const &t) { return continue_msg(); };

    /* vertices */

    // setup
    source_ptr = std::make_shared<source_node<Image>>(flow_graph, source_f, false);
    limiter_node<Image> limiter(flow_graph, image_parallel);
    broadcast_node<Image> raw_input_broadcast(flow_graph);

    // stage 1
    function_node<Image, f_node_result_t> max_node(flow_graph, 1, max_pixel_f);
    function_node<Image, f_node_result_t> min_node(flow_graph, 1, min_pixel_f);
    function_node<Image, f_node_result_t> search_node(flow_graph, 1, search_pixel_f);
    join_node<first_stage_tuple> stage1_joiner(flow_graph);
    broadcast_node<first_stage_tuple> image_with_selected_pixels_broadcast(flow_graph);


    // stage 2
    function_node<first_stage_tuple, bool> invert_border_f(flow_graph, unlimited, invert_selected_f);
    function_node<first_stage_tuple, bool> calc_average_f(flow_graph, unlimited, average_selected_f);

    join_node<tuple<bool, bool>> stage2_joiner(flow_graph);
    make_edge(invert_border_f, input_port<0>(stage2_joiner));
    make_edge(calc_average_f, input_port<1>(stage2_joiner));

    // tear down
    function_node<tuple<bool, bool>, continue_msg> decrement(flow_graph, unlimited, stub_continue);

    /* edges */

    // setup
    make_edge(*source_ptr, limiter);
    make_edge(raw_input_broadcast, max_node);

    // stage 1
    make_edge(raw_input_broadcast, max_node);
    make_edge(raw_input_broadcast, search_node);
    make_edge(raw_input_broadcast, min_node);
    make_edge(max_node, input_port<0>(stage1_joiner));
    make_edge(search_node, input_port<1>(stage1_joiner));
    make_edge(min_node, input_port<2>(stage1_joiner));
    make_edge(stage1_joiner, image_with_selected_pixels_broadcast);

    // stage 2
    make_edge(image_with_selected_pixels_broadcast, invert_border_f);
    make_edge(image_with_selected_pixels_broadcast, calc_average_f);
    make_edge(invert_border_f, input_port<0>(stage2_joiner));
    make_edge(calc_average_f, input_port<1>(stage2_joiner));

    // decrement
    make_edge(stage2_joiner, decrement);
    make_edge(decrement, limiter.decrement);
}

