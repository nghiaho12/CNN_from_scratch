#pragma once

#include <fstream>
#include <vector>
#include <cassert>

#include "cnn.hpp"

// Loader for the original MNIST dataset format
class MNIST_dataset {
public:
    MNIST_dataset(const char* image_path, const char* label_path) {
        images = load_images(image_path);
        labels = load_labels(label_path);

        assert(static_cast<int>(labels.size()) == images.shape[0]);
    }

    Tensor get_image(int idx) const {
        int h = images.shape[2];
        int w = images.shape[3];

        Tensor img(1, h, w);
        std::copy(&images.data[idx*h*w], &images.data[(idx+1)*h*w], img.data.begin());

        return img;
    }

    Tensor images;
    std::vector<uint8_t> labels;

private:
    Tensor load_images(const char* path) {
        std::ifstream is(path, std::ios::binary);
        if (!is) {
            throw std::runtime_error("can't open " + std::string(path));
        }

        char magic_str[4];
        char num_images_str[4];
        char num_rows_str[4];
        char num_cols_str[4];
        
        is.read(magic_str, 4);
        is.read(num_images_str, 4);
        is.read(num_rows_str, 4);
        is.read(num_cols_str, 4);

        // int magic = char4_to_int(magic_str); 
        int num_images = char4_to_int(num_images_str);
        int num_rows = char4_to_int(num_rows_str);
        int num_cols = char4_to_int(num_cols_str);

        assert(num_rows == 28);
        assert(num_cols == 28);

        std::vector<char> buf(num_images * num_rows * num_cols);
        is.read(buf.data(), buf.size());

        Tensor ret(num_images, 1, num_rows, num_cols);

        for (size_t i = 0; i < buf.size(); i++) {
            // norm to [0, 1]
            ret.data[i] = static_cast<uint8_t>(buf[i]) / 255.f;
        }

        return ret;
    }

    std::vector<uint8_t> load_labels(const char* path) {
        std::ifstream is(path, std::ios::binary);
        if (!is) {
            throw std::runtime_error("can't open " + std::string(path));
        }

        char magic_str[4];
        char num_str[4];
        
        is.read(magic_str, 4);
        is.read(num_str, 4);

        int num = char4_to_int(num_str);

        std::vector<uint8_t> ret(num);
        is.read(reinterpret_cast<char*>(ret.data()), ret.size());

        return ret;
    }

    int char4_to_int(char str[4]) {
        // MSB to LSB
        std::swap(str[0], str[3]);
        std::swap(str[1], str[2]);

        int num = *reinterpret_cast<int*>(str);

        return num;
    }
};

