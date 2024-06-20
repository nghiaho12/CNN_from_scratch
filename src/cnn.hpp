#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

class Tensor {
public:
    Tensor() = default; 

    Tensor(int size0) {
        shape = {size0};
        data.resize(size0);
    }

    Tensor(int size0, int size1) {
        shape = {size0, size1};
        data.resize(size0*size1);
    }

    Tensor(int size0, int size1, int size2) {
        shape = {size0, size1, size2};
        data.resize(size0*size1*size2);
    }

    Tensor(int size0, int size1, int size2, int size3) {
        shape = {size0, size1, size2, size3};
        data.resize(size0*size1*size2*size3);
    }

    float& operator()(int i) {
        assert(shape.size() == 1);
        assert(i < shape[0]);
        return data[i];
    }

    float operator()(int i) const {
        assert(shape.size() == 1);
        assert(i < shape[0]);
        return data[i];
    }

    float& operator()(int i, int j) {
        assert(shape.size() == 2);
        assert(i < shape[0]);
        assert(j < shape[1]);
        return data[i*shape[1] + j];
    }

    float operator()(int i, int j) const {
        assert(shape.size() == 2);
        assert(i < shape[0]);
        assert(j < shape[1]);
        return data[i*shape[1] + j];
    }

    float& operator()(int i, int j, int k) {
        assert(shape.size() == 3);
        assert(i < shape[0]);
        assert(j < shape[1]);
        assert(k < shape[2]);
        return data[i*shape[1]*shape[2] + j*shape[2] + k];
    }

    float operator()(int i, int j, int k) const {
        assert(shape.size() == 3);
        assert(i < shape[0]);
        assert(j < shape[1]);
        assert(k < shape[2]);
        return data[i*shape[1]*shape[2] + j*shape[2] + k];
    }

    float& operator()(int i, int j, int k, int l) {
        assert(shape.size() == 4);
        assert(i < shape[0]);
        assert(j < shape[1]);
        assert(k < shape[2]);
        assert(l < shape[3]);

        return data[i*shape[1]*shape[2]*shape[3] + j*shape[2]*shape[3] + k*shape[3] + l];
    }

    float operator()(int i, int j, int k, int l) const {
        assert(shape.size() == 4);
        assert(i < shape[0]);
        assert(j < shape[1]);
        assert(k < shape[2]);
        assert(l < shape[3]);

        return data[i*shape[1]*shape[2]*shape[3] + j*shape[2]*shape[3] + k*shape[3] + l];
    }

    template <typename RandGenerator>
    void set_random(double stdev, RandGenerator& gen) {
        std::normal_distribution dice(0.0, stdev);

        for (float &d: data) {
            d = dice(gen);
        }
    }

    void set_zero() {
        for (float &d: data) {
            d = 0.0;
        }
    }

    void set_one() {
        for (float &d: data) {
            d = 1.0;
        }
    }

    bool same_shape(const Tensor &rhs) const {
        if (shape.size() != rhs.shape.size()) {
            return false;
        }

        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] != rhs.shape[i]) {
                return false;
            }
        }

        return true;
    }

    Tensor operator* (float v) const {
        Tensor ret = *this;
        for (auto &x: ret.data) {
            x *= v;
        }
        return ret;
    }

    Tensor operator*= (float v) {
        for (auto &x: data) {
            x *= v;
        }
        return *this;
    }

    Tensor operator+ (const Tensor& rhs) const {
        Tensor ret = *this;
        for (size_t i = 0; i < rhs.data.size(); i++) {
            ret.data[i] += rhs.data[i];
        }
        return ret;
    }

    Tensor& operator+= (const Tensor& rhs) {
        for (size_t i = 0; i < rhs.data.size(); i++) {
            data[i] += rhs.data[i];
        }
        return *this;
    }

    Tensor operator- (const Tensor& rhs) {
        Tensor ret = *this;
        for (size_t i = 0; i < rhs.data.size(); i++) {
            ret.data[i] -= rhs.data[i];
        }
        return ret;
    }

    Tensor& operator-= (const Tensor& rhs) {
        for (size_t i = 0; i < rhs.data.size(); i++) {
            data[i] -= rhs.data[i];
        }
        return *this;
    }

    float sum() const {
        return std::accumulate(data.begin(), data.end(), 0.f);
    }

    friend std::ostream& operator<<(std::ostream& os, Tensor t) {
        int count = 0;

        os << "shape (";
        for (size_t i = 0; i < t.shape.size(); i++) {
            os << t.shape[i];

            if (i != t.shape.size() - 1) {
                os << ", ";
            }
        }
        os << ")\n";

        switch (t.shape.size()) {
            case 0: os << "shape (empty)"; break;
            case 1:
                for (auto d: t.data) {
                    os << d << " ";
                }
                os << "\n";
                break;
            case 2:
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        os << t.data[count] << " ";
                        count++;
                    }
                    os << "\n";
                }
                break;
            case 3:
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        for (int k = 0; k < t.shape[2]; k++) {
                            os << t.data[count] << " ";
                            count++;
                        }
                        os << "\n";
                    }
                    os << "\n";
                }

                break;
            case 4:
                for (int i = 0; i < t.shape[0]; i++) {
                    for (int j = 0; j < t.shape[1]; j++) {
                        for (int k = 0; k < t.shape[2]; k++) {
                            for (int l = 0; l < t.shape[3]; l++) {
                                os << t.data[count] << " ";
                                count++;
                            }
                            os << "\n";
                        }
                        os << "\n";
                    }
                }
                break;


            default: return os; break;
        }

        return os;
    }

    // exposed to make life easier
    std::vector<float> data;
    std::vector<int> shape;
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor operator()(const Tensor& in) = 0; 
    virtual Tensor backward(const Tensor& delta) = 0; 
    virtual void print(std::ostream& os) const = 0;

    friend std::ostream& operator<<(std::ostream& os, const Layer& rhs) {
        rhs.print(os);
        return os;
    }

    // optional
    Tensor weight;
    Tensor bias;

    int sum_count = 0;
    Tensor sum_weight_grad;
    Tensor sum_bias_grad;
    Tensor prev_weight_grad;
    Tensor prev_bias_grad;
};

class Conv2D : public Layer {
public:
    Conv2D(int in_channels, int out_channels, int ksize, int stride=1, int padding=0) {
        stride_ = stride;
        padding_ = padding;

        weight = Tensor(out_channels, in_channels, ksize, ksize);
        bias = Tensor(out_channels);

        sum_weight_grad = Tensor(out_channels, in_channels, ksize, ksize);
        sum_bias_grad = Tensor(out_channels);

        prev_weight_grad = Tensor(out_channels, in_channels, ksize, ksize);
        prev_bias_grad = Tensor(out_channels);
    }

    Tensor operator()(const Tensor& in) override {
        assert(in.shape[0] == weight.shape[1]);

        if (input_grad_.shape.empty()) {
            int out_h = new_out_dim(in.shape[1]);
            int out_w = new_out_dim(in.shape[2]);

            input_grad_ = Tensor(in.shape[0], in.shape[1], in.shape[2]);
            output_ = Tensor(weight.shape[0], out_w, out_h);
        }

        input_ = in;

        for (int oc = 0; oc < output_.shape[0]; oc++) {
            for (int oy = 0; oy < output_.shape[1]; oy++) {
                for (int ox = 0; ox < output_.shape[2]; ox++) {
                    float sum = 0;

                    for (int wc = 0; wc < weight.shape[1]; wc++) {
                        for (int wy = 0; wy < weight.shape[2]; wy++) {
                            for (int wx = 0; wx < weight.shape[3]; wx++) {
                               int in_y = oy*stride_ - padding_ + wy;
                               int in_x = ox*stride_ - padding_ + wx;

                               if (in_y < 0 || in_y >= in.shape[1]) {
                                  continue;
                               }
                               if (in_x < 0 || in_x >= in.shape[2]) {
                                  continue;
                               }

                               float x = in(wc, in_y, in_x);
                               float w = weight(oc, wc, wy, wx);

                               sum += w*x;
                            }
                        }
                    }

                    float b = bias(oc);
                    output_(oc, oy, ox) = sum + b;
                }
            }
        }

        return output_;
    }

    Tensor backward(const Tensor& delta) override {
        for (size_t i = 0; i < delta.shape.size(); i++) {
            assert(delta.shape[i] == output_.shape[i]);
        }

        input_grad_.set_zero();

        for (int oc = 0; oc < output_.shape[0]; oc++) {
            for (int oy = 0; oy < output_.shape[1]; oy++) {
                for (int ox = 0; ox < output_.shape[2]; ox++) {
                    float d = delta(oc, oy, ox);

                    // convolution
                    for (int wc = 0; wc < weight.shape[1]; wc++) {
                        for (int wy = 0; wy < weight.shape[2]; wy++) {
                            for (int wx = 0; wx < weight.shape[3]; wx++) {
                                int in_y = oy*stride_ - padding_ + wy;
                                int in_x = ox*stride_ - padding_ + wx;

                                if (in_y < 0 || in_y >= input_.shape[1]) {
                                   continue;
                                }
                                if (in_x < 0 || in_x >= input_.shape[2]) {
                                   continue;
                                }

                                float x = input_(wc, in_y, in_x);
                                float w = weight(oc, wc, wy, wx);

                                input_grad_(wc, in_y, in_x) += w*d;
                                sum_weight_grad(oc, wc, wy, wx) += x*d;
                            }
                        }
                    }

                    sum_bias_grad(oc) += d;
                }
            }
        }

        sum_count++;

        return input_grad_;
    }

    void print(std::ostream& os) const override { 
        os << "Conv2D (in=" << weight.shape[1] << " out=" << weight.shape[0] << " ksize=" << weight.shape[2] << " stride=" << stride_ << " padding=" << padding_ << ")";
    }

private:
    int new_out_dim(int x) {
        int ksize = weight.shape[2];
        float a = std::ceil(1.0*(x + 2*padding_ - ksize + 1) / stride_);
        return static_cast<int>(a);
    }

    int padding_;
    int stride_;
    int sum_count_ = 0;

    Tensor input_;
    Tensor input_grad_;
    Tensor output_;
};

class Dense : public Layer {
public:
    Dense(int in_size, int out_size) {
        weight = Tensor(out_size, in_size);
        bias = Tensor(out_size);

        sum_weight_grad = Tensor(out_size, in_size);
        sum_bias_grad = Tensor(out_size);

        prev_weight_grad = Tensor(out_size, in_size);
        prev_bias_grad = Tensor(out_size);

        output_ = Tensor(out_size);
        input_grad_ = Tensor(out_size, in_size);
    }

    Tensor operator()(const Tensor& in) override {
        assert(in.shape.size() == 1);
        assert(in.shape[0] == weight.shape[1]);

        input_ = in;

        for (int i = 0; i < weight.shape[0]; i++) {
            float sum = 0;
            for (int j = 0; j < weight.shape[1]; j++) {
                sum += weight(i, j) * in(j);
            }
            sum += bias(i);
            output_(i) = sum;
        }

        return output_;
    }

    Tensor backward(const Tensor& delta) override {
        for (int i = 0; i < weight.shape[0]; i++) {
            float d = delta(i);
            for (int j = 0; j < weight.shape[1]; j++) {
                input_grad_(i, j) = weight(i, j) * d;
                sum_weight_grad(i, j) += input_(j) * d;
            }
            sum_bias_grad(i) += d;
        }

        sum_count++;

        return input_grad_;
    }

    void print(std::ostream& os) const override { 
        os << "Dense (in=" << weight.shape[1] << " out=" << weight.shape[0] << ")";
    }

private:
    Tensor output_;
    Tensor input_;
    Tensor input_grad_;
};

class ReLU : public Layer {
public:
    Tensor operator()(const Tensor& in) override {
        if (output_.data.empty()) {
            output_ = in;
            input_grad_ = in;
        }

        int i = 0;
        for (float x : in.data) {
            output_.data[i] = std::max(0.f, x);
            i++;
        }

        return output_;
    }

    Tensor backward(const Tensor& delta) override {
        int i = 0;
        for (float x : output_.data) {
            if (x > 0) {
                input_grad_.data[i] = delta.data[i]; 
            } else {
                input_grad_.data[i] = 0;
            }

            i++;
        }

        return input_grad_;
    }

    void print(std::ostream& os) const override { 
        os << "ReLU";
    }

private:
    Tensor output_;
    Tensor input_grad_;
};

class Flatten : public Layer {
public:
    Tensor operator()(const Tensor& in) override {
        if (output_.shape.empty()) {
            output_ = Tensor(in.data.size());
        }

        input_ = in;
        output_.data = in.data;

        return output_;
    }

    Tensor backward(const Tensor& delta) override {
        assert(input_.data.size() == delta.data.size());

        input_.data = delta.data;
        return input_;
    }

    void print(std::ostream& os) const override { 
        os << "Flatten";
    }

private:
    Tensor input_;
    Tensor output_;
};

class Softmax : public Layer {
public:
    Tensor operator()(const Tensor& in) override {
        assert(in.shape.size() == 1);

        output_ = in;

        // sum of exp trick for numerical stability
        float max_val = *std::max_element(in.data.begin(), in.data.end());

        float sum_exp = 0;
        for (auto x : in.data) {
            sum_exp += std::exp(x - max_val);
        }

        int i = 0;
        for (auto x : in.data) {
            output_.data[i] = std::exp(x - max_val) / sum_exp;
            i++;
        }

        return output_;
    }

    Tensor backward(const Tensor& delta) override {
        Tensor ret = output_;

        for (int i = 0; i < output_.shape[0]; i++) { 
            float sum = 0;
            for (int j = 0; j < output_.shape[0]; j++) {
                if (i == j) {
                    sum += output_(i)*(1 - output_(i)) * delta(j);
                } else {
                    sum += -output_(i)*output_(j) * delta(j);
                }
            }
            ret(i) = sum;
        }

        return ret;
    }

    void print(std::ostream& os) const override { 
        os << "Softmax";
    }

private:
    Tensor output_;
};

class CrossEntropyLoss {
public:
    float operator()(const Tensor& y, int target) {
        assert(y.shape.size() == 1);

        y_ = y;
        target_ = target;
      
        float sum = 0;

        for (size_t i = 0; i < y_.data.size(); i++) {
            if (static_cast<int>(i) == target_) {
                sum += -std::log(std::max(y.data[i], EPS));
            } else {
                sum += -std::log(std::max(1 - y.data[i], EPS));
            }
        }

        return sum;
    }

    Tensor backward() {
        Tensor ret = y_;

        for (size_t i = 0; i < y_.data.size(); i++) {
            if (static_cast<int>(i) == target_) {
                ret.data[i] = -1.0/std::max(y_.data[i], EPS); 
            } else {
                ret.data[i] = 1.0/std::max(1 - y_.data[i], EPS); 
            }
        }

        return ret;
    }

private:
    Tensor y_;
    int target_;
    const float EPS = std::numeric_limits<float>::epsilon();
};

class AccuracyMetric {
public:
    AccuracyMetric(int num_classes) {
        confusion_ = Tensor(num_classes, num_classes);
    }

    void update(Tensor y, int target) {
        assert(y.shape[0] == confusion_.shape[0]);

        float m = y(0);
        int pred = 0;

        for (int i = 1; i < y.shape[0]; i++) {
            if (y(i) > m) {
                m = y(i);
                pred = i;
            }
        }

        assert(pred >= 0);

        confusion_(target, pred)++;
        total_++;
    }

    float accuracy() {
        int correct = 0;
        for (int i = 0; i < confusion_.shape[0]; i++) {
            correct += confusion_(i, i);
        }

        return 1.0f*correct / total_;
    }

    void print_confusion_matrix() { 
        for (int i = 0; i < confusion_.shape[0]; i++) {
            for (int j = 0; j < confusion_.shape[0]; j++) {
                std::cout << std::setw(5) << confusion_(i, j);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        std::cout << "rows = ground truth\n";
        std::cout << "cols = predicted\n";
    }

    void clear() {
        confusion_.set_zero();
        total_ = 0;
    }

private:
    Tensor confusion_;
    int total_ = 0;

};

template <typename RandGenerator>
void init_network_weight(std::vector<Layer*> &net, RandGenerator& gen) {
    for (auto layer : net) {
        layer->bias.set_zero();
        auto s = layer->weight.shape;
        int fan_in = 0;

        if (s.size() == 4) { // Conv2D
            fan_in = s[1] * s[2] * s[3];
        } else if (s.size() == 2) { // Dense
            fan_in = layer->weight.shape[1];
        }

        if (fan_in > 0) {
            float stdev = std::sqrt(1.f / fan_in);
            layer->weight.set_random(stdev, gen);
        }
    }
}

void SGD_weight_update(std::vector<Layer*> &net, float lr, float momentum) {
    for (auto l : net) {
        if (l->weight.shape.empty()) {
            continue;
        }

        // apply momentum
        Tensor weight_grad = l->sum_weight_grad * (1.f/l->sum_count) + l->prev_weight_grad*momentum;
        Tensor bias_grad = l->sum_bias_grad * (1.f/l->sum_count) + l->prev_bias_grad*momentum;

        // gradient descent
        l->weight -= weight_grad*lr; 
        l->bias -= bias_grad*lr;

        l->prev_weight_grad = weight_grad;
        l->prev_bias_grad = bias_grad;

        l->sum_weight_grad.set_zero();
        l->sum_bias_grad.set_zero();
        l->sum_count = 0;
    }
}

void print_network_info(const std::vector<Layer*> &net) {
    std::cout << "network layers:\n";

    int total_params = 0;
    int i = 0;
    for (auto layer : net) {
        std::cout << "  " << i << ": " << *layer << "\n";
        total_params += layer->weight.data.size() + layer->bias.data.size();
        i++;
    }

    std::cout << "\ntotal trainable params: " << total_params << "\n";
}
