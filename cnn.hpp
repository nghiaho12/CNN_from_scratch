#include <iostream>
#include <fstream>

#include <ostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

std::default_random_engine gen;

class Tensor {
public:
    Tensor() = default; 

    Tensor(int size0) {
        shape_ = {size0};
        data_.resize(size0);
    }

    Tensor(int size0, int size1) {
        shape_ = {size0, size1};
        data_.resize(size0*size1);
    }

    Tensor(int size0, int size1, int size2) {
        shape_ = {size0, size1, size2};
        data_.resize(size0*size1*size2);
    }

    Tensor(int size0, int size1, int size2, int size3) {
        shape_ = {size0, size1, size2, size3};
        data_.resize(size0*size1*size2*size3);
    }

    float& operator()(int i) {
        assert(shape_.size() == 1);
        assert(i < shape_[0]);
        return data_[i];
    }

    float& operator()(int i, int j) {
        assert(shape_.size() == 2);
        assert(i < shape_[0]);
        assert(j < shape_[1]);
        return data_[i*shape_[1] + j];
    }

    float& operator()(int i, int j, int k) {
        assert(shape_.size() == 3);
        assert(i < shape_[0]);
        assert(j < shape_[1]);
        assert(k < shape_[2]);
        return data_[i*shape_[1]*shape_[2] + j*shape_[2] + k];
    }

    float& operator()(int i, int j, int k, int l) {
        assert(shape_.size() == 4);
        assert(i < shape_[0]);
        assert(j < shape_[1]);
        assert(k < shape_[2]);
        assert(l < shape_[3]);

        return data_[i*shape_[1]*shape_[2]*shape_[3] + j*shape_[2]*shape_[3] + k*shape_[3] + l];
    }

    int shape(int dim) const { return shape_.at(dim); }
    std::vector<int> shape() const { return shape_; }

    void set_random() {
        std::normal_distribution dice(0.0, 1.0);

        for (float &d: data_) {
            d = dice(gen);
        }
    }

    void set_zero() {
        for (float &d: data_) {
            d = 0.0;
        }
    }

    void set_one() {
        for (float &d: data_) {
            d = 1.0;
        }
    }

    Tensor operator* (float v) {
        Tensor ret = *this;
        for (auto &x: ret.data_) {
            x *= v;
        }
        return ret;
    }

    Tensor operator- (const Tensor& rhs) {
        Tensor ret = *this;
        for (size_t i = 0; i < rhs.data_.size(); i++) {
            ret.data_[i] -= rhs.data_[i];
        }
        return ret;
    }

    Tensor operator-= (const Tensor& rhs) {
        for (size_t i = 0; i < rhs.data_.size(); i++) {
            data_[i] -= rhs.data_[i];
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, Tensor t) {
        int count = 0;

        os << "shape (";
        for (size_t i = 0; i < t.shape().size(); i++) {
            os << t.shape(i);

            if (i != t.shape().size() - 1) {
                os << ", ";
            }
        }
        os << ")\n";

        switch (t.shape().size()) {
            case 0: os << "shape (empty)"; break;
            case 1:
                for (auto d: t.data_) {
                    os << d << " ";
                }
                os << "\n";
                break;
            case 2:
                for (int i = 0; i < t.shape(0); i++) {
                    for (int j = 0; j < t.shape(1); j++) {
                        os << t.data_[count] << " ";
                        count++;
                    }
                    os << "\n";
                }
                break;
            case 3:
                for (int i = 0; i < t.shape(0); i++) {
                    for (int j = 0; j < t.shape(1); j++) {
                        for (int k = 0; k < t.shape(2); k++) {
                            os << t.data_[count] << " ";
                            count++;
                        }
                        os << "\n";
                    }
                    os << "\n";
                }

                break;
            case 4:
                for (int i = 0; i < t.shape(0); i++) {
                    for (int j = 0; j < t.shape(1); j++) {
                        for (int k = 0; k < t.shape(2); k++) {
                            for (int l = 0; l < t.shape(3); l++) {
                                os << t.data_[count] << " ";
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

    std::vector<float> data_;

private:
    std::vector<int> shape_;
};

class Layer {
public:
    virtual Tensor operator()(Tensor in) = 0; 
    virtual Tensor backward(Tensor delta) = 0; 
    virtual void zero() = 0;
    virtual void update_weight(float lr) = 0;
    virtual std::string name() = 0;
};

class Conv2D : public Layer {
public:
    Conv2D(int in_channels, int out_channels, int ksize, int stride=1, int padding=0) {
        in_channels_ = in_channels;
        out_channels_ = out_channels;
        ksize_ = ksize;
        stride_ = stride;
        padding_ = padding;

        weight_ = Tensor(out_channels, in_channels, ksize, ksize);
        weight_.set_random();
        dweight_ = Tensor(out_channels, in_channels, ksize, ksize);

        bias_ = Tensor(out_channels);
        bias_.set_random();
        dbias_ = Tensor(out_channels);
    }
    
    std::string name() override {
        return "Conv2D";
    }

    Tensor operator()(Tensor in) override {
        assert(in.shape(0) == in_channels_);

        if (dinput_.shape().empty()) {
            int out_h = new_out_dim(in.shape(1));
            int out_w = new_out_dim(in.shape(2));

            dinput_ = Tensor(in.shape(0), in.shape(1), in.shape(2));
            output_ = Tensor(out_channels_, out_w, out_h);
        }

        input_ = in;

        for (int oc = 0; oc < output_.shape(0); oc++) {
            for (int oy = 0; oy < output_.shape(1); oy++) {
                for (int ox = 0; ox < output_.shape(2); ox++) {

                    float sum = 0;

                    for (int kc = 0; kc < in_channels_; kc++) {
                        for (int ky = 0; ky < ksize_; ky++) {
                            for (int kx = 0; kx < ksize_; kx++) {
                                int in_y = oy*stride_ - padding_ + ky;
                                int in_x = ox*stride_ - padding_ + kx;

                                if (in_y < 0 || in_y >= in.shape(1)) {
                                   continue;
                                }
                                if (in_x < 0 || in_x >= in.shape(2)) {
                                   continue;
                                }

                                float x = in(kc, in_y, in_x);
                                float w = weight_(oc, kc, ky, kx);

                                sum += w*x;
                            }
                        }
                    }

                    float b = bias_(oc);
                    output_(oc, oy, ox) = sum + b;
                }
            }
        }

        return output_;
    }

    Tensor backward(Tensor delta) override {
        for (size_t i = 0; i < delta.shape().size(); i++) {
            assert(delta.shape(i) == output_.shape(i));
        }

        dinput_.set_zero();

        for (int oc = 0; oc < output_.shape(0); oc++) {
            for (int oy = 0; oy < output_.shape(1); oy++) {
                for (int ox = 0; ox < output_.shape(2); ox++) {
                    float d = delta(oc, oy, ox);

                    // convolution
                    for (int kc = 0; kc < in_channels_; kc++) {
                        for (int ky = 0; ky < ksize_; ky++) {
                            for (int kx = 0; kx < ksize_; kx++) {
                                int in_y = oy*stride_ - padding_ + ky;
                                int in_x = ox*stride_ - padding_ + kx;

                                if (in_y < 0 || in_y >= input_.shape(1)) {
                                   continue;
                                }
                                if (in_x < 0 || in_x >= input_.shape(2)) {
                                   continue;
                                }

                                float x = input_(kc, in_y, in_x);
                                float w = weight_(oc, kc, ky, kx);

                                dinput_(kc, in_y, in_x) += w*d;
                                dweight_(oc, kc, ky, kx) += x*d;
                            }
                        }
                    }

                    dbias_(oc) += d;
                }
            }
        }

        count_++;

        return dinput_;
    }

    void zero() override {
        dweight_.set_zero();
        dbias_.set_zero();
        count_ = 0;
    }

    void update_weight(float lr) override {
        weight_ -= dweight_ * (lr / count_);
        bias_ -= dbias_ * (lr / count_);
    } 

    friend std::ostream& operator<<(std::ostream& os, Conv2D t) {
        os << "Conv2D (ksize=" << t.ksize_ << " padding=" << t.padding_ << " stride=" << t.stride_ << ")\n";
        os << "weight_: " << t.weight_ << "\n";
        os << "bias_: " << t.bias_ << "\n";
        os << "dweight_: " << t.dweight_ << "\n";
        os << "dbias_: " << t.dbias_ << "\n";
        os << "dinput: " << t.dinput_ << "\n";

        return os;
    }

private:
    int new_out_dim(int x) {
        float a = std::ceil(1.0*(x + 2*padding_ - ksize_ + 1) / stride_);
        return static_cast<int>(a);
    }

    int ksize_;
    int padding_;
    int stride_;
    int in_channels_;
    int out_channels_;
    int count_ = 0;

    Tensor weight_;
    Tensor dweight_;
    Tensor bias_;
    Tensor dbias_;
    Tensor input_;
    Tensor dinput_;
    Tensor output_;
};

class ReLU: public Layer {
public:
    Tensor operator()(Tensor in) override {
        output_ = in;

        for (auto& x : output_.data_) {
            x = std::max(0.f, x);
        }

        return output_;
    }

    Tensor backward(Tensor delta) override {
        dinput_ = output_;

        for (size_t i = 0; i < dinput_.data_.size(); i++) {
            if (output_.data_[i] > 0) {
                dinput_.data_[i] = delta.data_[i]; 
            } else {
                dinput_.data_[i] = 0;
            }
        }

        return dinput_;
    }

    std::string name() override {
        return "ReLU";
    }

    void zero() override {} 
    void update_weight(float lr) override {} 

private:
    Tensor output_;
    Tensor dinput_;
};

class Flatten: public Layer {
public:
    Tensor operator()(Tensor in) override {
        if (output_.shape().empty()) {
            output_ = Tensor(in.data_.size());
        }

        input_ = in;
        output_.data_ = in.data_;

        return output_;
    }

    Tensor backward(Tensor delta) override {
        assert(input_.data_.size() == delta.data_.size());

        input_.data_ = delta.data_;
        return input_;
    }

    std::string name() override {
        return "Flatten";
    }

    void zero() override {} 
    void update_weight(float lr) override {} 

private:
    Tensor input_;
    Tensor output_;
};

class Softmax: public Layer {
public:
    Tensor operator()(Tensor in) override {
        output_ = in;

        float m = in(0);
        float sum_exp = 0;

        for (auto a: in.data_) {
            m = std::max(m, a);
        }

        for (auto a: in.data_) {
            sum_exp += std::exp(a - m);
        }

        for (size_t i = 0; i < in.data_.size(); i++) {
            output_.data_[i] = std::exp(in.data_[i] - m) / sum_exp;
        }

        return output_;
    }

    Tensor backward(Tensor delta) override {
        Tensor ret = output_;

        for (int i = 0; i < output_.shape(0); i++) {
            ret(i) = output_(i)*(1 - output_(i)) * delta(i);
        }

        return ret;
    }

    std::string name() override {
        return "Softmax";
    }

    void zero() override {} 
    void update_weight(float lr) override {} 

private:
    Tensor output_;
};

// For multi-class
class CrossEntropyLoss {
public:
    float operator()(Tensor y, int target) {
        y_ = y;
        target_ = target;
      
        float sum = 0;
        float eps = 1e-6;

        for (int i = 0; i < static_cast<int>(y.data_.size()); i++) {
            if (i == target) {
                sum += -std::log(y.data_[i] + eps);
            } else {
                sum += -std::log(1 - y.data_[i] + eps);
            }
        }

        return sum;
    }

    Tensor backward() {
        Tensor ret = y_;

        for (int i = 0; i < static_cast<int>(y_.data_.size()); i++) {
            if (i == target_) {
                ret.data_[i] = -1.0/y_.data_[i]; 
            } else {
                ret.data_[i] = 1.0/(1 - y_.data_[i]); 
            }
        }

        return ret;
    }

private:
    Tensor y_;
    int target_;
};

class MNIST {
public:
    MNIST(const char* train_images, const char* train_labels, const char* test_images, const char* test_labels) {
        train_ = load_images(train_images);
        test_ = load_images(test_images);

        train_label_ = load_labels(train_labels);
        test_label_ = load_labels(test_labels);

        assert(static_cast<int>(train_label_.size()) == train_.shape(0));
        assert(static_cast<int>(test_label_.size()) == test_.shape(0));
    }

    Tensor get_train_image(int idx) const {
        int h = train_.shape(2);
        int w = train_.shape(3);

        Tensor img(1, h, w);
        std::copy(&train_.data_[idx*h*w], &train_.data_[(idx+1)*h*w], img.data_.begin());

        return img;
    }

    uint8_t get_train_label(int idx) const {
        return train_label_[idx];
    }


    Tensor get_test_image(int idx) const {
        int h = test_.shape(2);
        int w = test_.shape(3);

        Tensor img(1, h, w);
        std::copy(&test_.data_[idx*h*w], &test_.data_[(idx+1)*h*w], img.data_.begin());

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
            ret.data_[i] = static_cast<uint8_t>(buf[i]) / 255.f;
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

