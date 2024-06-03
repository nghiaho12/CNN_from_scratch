#include <iostream>
#include <fstream>

#include <ostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

std::default_random_engine fixed_random_gen{42};

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

    void set_random(double stdev) {
        std::normal_distribution dice(0.0, stdev);

        for (float &d: data) {
            d = dice(fixed_random_gen);
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

    Tensor operator* (float v) {
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

    Tensor operator+ (const Tensor& rhs) {
        Tensor ret = *this;
        for (size_t i = 0; i < rhs.data.size(); i++) {
            ret.data[i] += rhs.data[i];
        }
        return ret;
    }

    Tensor operator+= (const Tensor& rhs) {
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

    Tensor operator-= (const Tensor& rhs) {
        for (size_t i = 0; i < rhs.data.size(); i++) {
            data[i] -= rhs.data[i];
        }
        return *this;
    }

    float sum() {
        float s = 0;
        for (float x: data) {
            s += x;
        }
        return s;
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
    virtual std::string name() = 0;
    virtual Tensor operator()(const Tensor& in) = 0; 
    virtual Tensor backward(const Tensor& delta) = 0; 

    // used by Conv2D and Dense
    Tensor weight;
    Tensor bias;

    // Tensor accumulates for every backward() call
    Tensor dweight; // derivative of output with respect to weight
    Tensor dbias;// derivative of output with respect to bias
    int batch_count = 0; // number of backward() calls

    Tensor prev_dweight;
    Tensor prev_dbias;
};

class Conv2D : public Layer {
public:
    std::string name() override {
        return "Conv2D";
    }

    Conv2D(int in_channels, int out_channels, int ksize, int stride=1, int padding=0) {
        stride_ = stride;
        padding_ = padding;

        weight = Tensor(out_channels, in_channels, ksize, ksize);
        dweight = Tensor(out_channels, in_channels, ksize, ksize);
        prev_dweight = Tensor(out_channels, in_channels, ksize, ksize);

        bias = Tensor(out_channels);
        dbias = Tensor(out_channels);
        prev_dbias = Tensor(out_channels);
    }

    Tensor operator()(const Tensor& in) override {
        assert(in.shape[0] == weight.shape[1]);

        if (dinput_.shape.empty()) {
            int out_h = new_out_dim(in.shape[1]);
            int out_w = new_out_dim(in.shape[2]);

            dinput_ = Tensor(in.shape[0], in.shape[1], in.shape[2]);
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

        dinput_.set_zero();

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

                                dinput_(wc, in_y, in_x) += w*d;
                                dweight(oc, wc, wy, wx) += x*d;
                            }
                        }
                    }

                    dbias(oc) += d;
                }
            }
        }

        batch_count++;

        return dinput_;
    }

    friend std::ostream& operator<<(std::ostream& os, Conv2D t) {
        os << "Conv2D (ksize=" << t.weight.shape[2] << " padding=" << t.padding_ << " stride=" << t.stride_ << ")\n";
        os << "weight: " << t.weight << "\n";
        os << "bias: " << t.bias << "\n";
        os << "dweight: " << t.dweight << "\n";
        os << "dbias: " << t.dbias << "\n";
        os << "dinput: " << t.dinput_ << "\n";

        return os;
    }

private:
    int new_out_dim(int x) {
        int ksize = weight.shape[2];
        float a = std::ceil(1.0*(x + 2*padding_ - ksize + 1) / stride_);
        return static_cast<int>(a);
    }

    int padding_;
    int stride_;
    int count_ = 0;

    Tensor input_;
    Tensor dinput_;
    Tensor output_;
};

class ReLU: public Layer {
public:
    Tensor operator()(const Tensor& in) override {
        if (output_.data.empty()) {
            output_ = in;
        }

        size_t i = 0;
        for (float x : in.data) {
            output_.data[i] = std::max(0.f, x);
            i++;
        }

        return output_;
    }

    Tensor backward(const Tensor& delta) override {
        if (dinput_.data.empty()) {
            dinput_ = output_;
        }

        size_t i = 0;
        for (float x : output_.data) {
            if (x > 0) {
                dinput_.data[i] = delta.data[i]; 
            } else {
                dinput_.data[i] = 0;
            }

            i++;
        }

        return dinput_;
    }

    std::string name() override {
        return "ReLU";
    }

private:
    Tensor output_;
    Tensor dinput_;
};

class Flatten: public Layer {
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

    std::string name() override {
        return "Flatten";
    }

private:
    Tensor input_;
    Tensor output_;
};

class Softmax: public Layer {
public:
    Tensor operator()(const Tensor& in) override {
        output_ = in;

        // sum of exp trick for numerical stability
        float m = in(0);
        float sum_exp = 0;

        // find max value
        for (auto x : in.data) {
            m = std::max(m, x);
        }

        // subtract max value
        for (auto x : in.data) {
            sum_exp += std::exp(x - m);
        }

        size_t i = 0;
        for (auto x : in.data) {
            output_.data[i] = std::exp(x - m) / sum_exp;
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

    std::string name() override {
        return "Softmax";
    }

private:
    Tensor output_;
};

// For multi-class
class CrossEntropyLoss {
public:
    float operator()(const Tensor& y, int target) {
        y_ = y;
        target_ = target;
      
        float sum = 0;

        for (int i = 0; i < static_cast<int>(y.data.size()); i++) {
            if (i == target) {
                sum += -std::log(std::max(y.data[i], EPS));
            } else {
                sum += -std::log(std::max(1 - y.data[i], EPS));
            }
        }

        return sum;
    }

    Tensor backward() {
        Tensor ret = y_;

        for (int i = 0; i < static_cast<int>(y_.data.size()); i++) {
            if (i == target_) {
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

    Tensor confusion_matrix() { 
        return confusion_;
    }

    void clear() {
        confusion_.set_zero();
        total_ = 0;
    }

private:
    Tensor confusion_;
    int total_ = 0;

};

class MNIST {
public:
    MNIST(const char* train_images, const char* train_labels, const char* test_images, const char* test_labels) {
        train_ = load_images(train_images);
        test_ = load_images(test_images);

        train_label_ = load_labels(train_labels);
        test_label_ = load_labels(test_labels);

        assert(static_cast<int>(train_label_.size()) == train_.shape[0]);
        assert(static_cast<int>(test_label_.size()) == test_.shape[0]);
    }

    Tensor get_train_image(int idx) const {
        int h = train_.shape[2];
        int w = train_.shape[3];

        Tensor img(1, h, w);
        std::copy(&train_.data[idx*h*w], &train_.data[(idx+1)*h*w], img.data.begin());

        return img;
    }

    uint8_t get_train_label(int idx) const {
        return train_label_[idx];
    }


    Tensor get_test_image(int idx) const {
        int h = test_.shape[2];
        int w = test_.shape[3];

        Tensor img(1, h, w);
        std::copy(&test_.data[idx*h*w], &test_.data[(idx+1)*h*w], img.data.begin());

        return img;
    }
   
    uint8_t get_test_label(int idx) const {
        return test_label_[idx];
    }


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

    Tensor train_;
    Tensor test_;
    std::vector<uint8_t> train_label_;
    std::vector<uint8_t> test_label_;
};

void init_weight_kaiming_he(std::vector<Layer*> &net) {
    for (Layer* layer: net) {
        if (layer->name() == "Conv2D") {
            layer->bias.set_zero();

            auto s = layer->weight.shape;
            int fan_in = s[1] * s[2] * s[3];
            float stdev = std::sqrt(1.f / fan_in);
            layer->weight.set_random(stdev);
        }
    }
}

void SGD_weight_update(std::vector<Layer*> &net, float lr, float momentum) {
    for (Layer* layer: net) {
        if (layer->batch_count == 0) {
            continue;
        }

        // average the weights
        layer->dweight *= 1.f/layer->batch_count;
        layer->dbias *= 1.f/layer->batch_count;

        // apply momentum
        layer->dweight += layer->prev_dweight*momentum;
        layer->dbias += layer->prev_dbias*momentum;

        // gradient descent
        layer->weight -= layer->dweight*lr; 
        layer->bias -= layer->dbias*lr;

        layer->prev_dweight = layer->dweight;
        layer->prev_dbias = layer->dbias;

        // zero out the gradients
        layer->dweight.set_zero();
        layer->dbias.set_zero();
        layer->batch_count = 0;
    }
}
