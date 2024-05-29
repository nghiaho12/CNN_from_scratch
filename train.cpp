#include <iostream>
#include <fstream>
#include <ostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

std::default_random_engine gen;

constexpr int WIDTH = 28;
constexpr int HEIGHT = 28;
constexpr int IMAGE_SIZE = WIDTH*HEIGHT;
constexpr int HEADER_SIZE = 12;

struct Tensor {
    Tensor() {
        dim = 0;
    }

    Tensor(int size) {
        dim = 1;
        s0 = size;
        data.resize(s0);
    }

    Tensor(int size0, int size1) {
        dim = 2;
        s0 = size0;
        s1 = size1;
        data.resize(s0*s1);
    }

    Tensor(int size0, int size1, int size2) {
        dim = 3;
        s0 = size0;
        s1 = size1;
        s2 = size2;
        data.resize(s0*s1*s2);
    }

    Tensor(int size0, int size1, int size2, int size3) {
        dim = 4;
        s0 = size0;
        s1 = size1;
        s2 = size2;
        s3 = size3;
        data.resize(s0*s1*s2*s3);
    }

    void set_random() {
        std::normal_distribution dice(0.0, 1.0);

        for (float &d: data) {
            d = dice(gen);
        }
    }

    void set_zero() {
        for (float &d: data) {
            d = 0;
        }
    }

    float& operator()(int i) {
        assert(dim == 1);
        assert(i < s0);
        return data[i];
    }

    float& operator()(int i, int j) {
        assert(dim == 2);
        assert(i < s0);
        assert(j < s1);
        return data[i*s1 + j];
    }

    float& operator()(int i, int j, int k) {
        assert(dim == 3);
        assert(i < s0);
        assert(j < s1);
        assert(k < s2);
        return data[i*s0*s1 + j*s1 + k];
    }

    float& operator()(int i, int j, int k, int l) {
        assert(dim == 4);
        assert(i < s0);
        assert(j < s1);
        assert(k < s2);
        assert(l < s3);

        return data[i*s0*s1*s2 + j*s1*s2 + k*s2 + l];
    }

    Tensor operator* (float v) {
        Tensor ret = *this;
        for (auto &x: ret.data) {
            x *= v;
        }
        return ret;
    }

    Tensor operator- (const Tensor& rhs) {
        Tensor ret = *this;
        for (size_t i = 0; i < rhs.data.size(); i++) {
            ret.data[i] -= rhs.data[i];
        }
        return ret;
    }


    friend std::ostream& operator<<(std::ostream& os, Tensor t) {
        int count = 0;

        switch (t.dim) {
            case 0: os << "shape (empty)"; break;
            case 1:
                os << "shape (" << t.s0 << ")\n";
                for (auto d: t.data) {
                    os << d << " ";
                }
                os << "\n";
                break;
            case 2:
                os << "shape (" << t.s0 << ", " << t.s1 << ")\n";

                for (int i = 0; i < t.s0; i++) {
                    for (int j = 0; j < t.s1; j++) {
                        os << t.data[count] << " ";
                        count++;
                    }
                    os << "\n";
                }
                break;
            case 3:
                os << "shape (" << t.s0 << ", " << t.s1 << ", " << t.s2 << ")\n";

                for (int i = 0; i < t.s0; i++) {
                    for (int j = 0; j < t.s1; j++) {
                        for (int k = 0; k < t.s2; k++) {

                            os << t.data[count] << " ";
                            count++;
                        }
                        os << "\n";
                    }
                }

                break;
            case 4:
                os << "shape (" << t.s0 << ", " << t.s1 << ", " << t.s2 << ", " << t.s3 << ")\n";

                for (int i = 0; i < t.s0; i++) {
                    for (int j = 0; j < t.s1; j++) {
                        for (int k = 0; k < t.s2; k++) {
                            for (int l = 0; l < t.s3; l++) {

                                os << t.data[count] << " ";
                                count++;
                            }
                            os << "\n";
                        }
                    }
                }
                break;


            default: return os; break;
        }

        return os;
    }

    int dim;
    int s0;
    int s1;
    int s2;
    int s3;

    std::vector<float> data;
};

class Layer {
public:
    virtual Tensor operator()(Tensor in) = 0;
    virtual Tensor backward(Tensor delta) = 0;
    virtual void zero() = 0;
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

    Tensor operator()(Tensor in) override {
        if (!init_) {
            int out_h = new_out_dim(in.s1);
            int out_w = new_out_dim(in.s2);

            dinput_ = Tensor(in.s0, in.s1, in.s2);
            output_ = Tensor(out_channels_, out_w, out_h);

            init_ = true;
        }

        input_ = in;

        for (int oc = 0; oc < output_.s0; oc++) {
            for (int oy = 0; oy < output_.s1; oy++) {
                for (int ox = 0; ox < output_.s2; ox++) {

                    float sum = 0;

                    for (int kc = 0; kc < in_channels_; kc++) {
                        for (int ky = 0; ky < ksize_; ky++) {
                            for (int kx = 0; kx < ksize_; kx++) {
                                int in_y = oy*stride_ - padding_ + ky;
                                int in_x = ox*stride_ - padding_ + kx;

                                if (in_y < 0 || in_y >= in.s1) {
                                   continue;
                                }
                                if (in_x < 0 || in_x >= in.s2) {
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
        assert(delta.s0 == output_.s0);
        assert(delta.s1 == output_.s1);
        assert(delta.s2 == output_.s2);
        assert(delta.s3 == output_.s3);

        dinput_.set_zero();
        dweight_.set_zero();
        dbias_.set_zero();

        for (int oc = 0; oc < output_.s0; oc++) {
            for (int oy = 0; oy < output_.s1; oy++) {
                for (int ox = 0; ox < output_.s2; ox++) {
                    float d = delta(oc, oy, ox);
                    // convolution
                    for (int kc = 0; kc < in_channels_; kc++) {
                        for (int ky = 0; ky < ksize_; ky++) {
                            for (int kx = 0; kx < ksize_; kx++) {
                                int in_y = oy*stride_ - padding_ + ky;
                                int in_x = ox*stride_ - padding_ + kx;

                                if (in_y < 0 || in_y >= input_.s1) {
                                   continue;
                                }
                                if (in_x < 0 || in_x >= input_.s2) {
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

        return dinput_;
    }

    void zero() override {
        dweight_.set_zero();
        dbias_.set_zero();
        dinput_.set_zero();
    }

    friend std::ostream& operator<<(std::ostream& os, Conv2D t) {
        os << "Conv2D (ksize=" << t.ksize_ << " padding=" << t.padding_ << " stride=" << t.stride_ << ")\n";
        os << "weight: " << t.weight_ << "\n";
        os << "bias: " << t.bias_ << "\n";
        os << "dweight: " << t.dweight_ << "\n";
        os << "dbias: " << t.dbias_ << "\n";
        os << "dinput: " << t.dinput_ << "\n";

        return os;
    }

    int new_out_dim(int x) {
        float a = std::ceil(1.0*(x + 2*padding_ - ksize_ + 1) / stride_);
        return static_cast<int>(a);
    }

    bool init_ = false;
    int ksize_;
    int padding_;
    int stride_;
    int in_channels_;
    int out_channels_;

    Tensor weight_;
    Tensor dweight_;
    Tensor bias_;
    Tensor dbias_;
    Tensor input_;
    Tensor dinput_;
    Tensor output_;
};

class Softmax : public Layer {
public:
    Tensor operator()(Tensor in) override {
        output_ = in;

        float m = in.data[0];
        float sum_exp = 0;

        for (auto a: in.data) {
            m = std::max(m, a);
        }

        for (auto a: in.data) {
            sum_exp += std::exp(a - m);
        }

        for (size_t i = 0; i < in.data.size(); i++) {
            output_.data[i] = std::exp(in.data[i] - m) / sum_exp;
        }

        return output_;
    }

    Tensor backward(Tensor delta) override {
        Tensor ret = output_;

        // expect only one value > 0
        int index = -1;

        for (size_t i = 0; i < delta.data[i]; i++) {
            if (delta.data[i] > 0) {
                index = i;
                break;
            }
        }

        assert(index >= 0);

        for (size_t i = 0; i < output_.data.size(); i++) {
            if (static_cast<int>(i) == index) {
                ret.data[i] = output_.data[i]*(1 - output_.data[i]);                
            } else {
                ret.data[i] = -output_.data[i]*output_.data[index];
            }
        }

        return ret;
    }

    void zero() override {}

    Tensor output_;
};

// For multi-class
class CrossEntropyLoss {
public:
    Tensor operator()(Tensor y, int target) {
        y_ = y;
        target_ = target;
        
        return -std::log(y.data[target]);
    }

    Tensor backward() {
        Tensor ret_ = y_;
        ret_.set_zero();

        ret_.data[target_] = 1.0/y_.data[target_];

        return ret_;
    }

    Tensor y_;
    int target_;
};

class MSELoss {
public:
    float operator()(Tensor in, Tensor target) {
        assert(in.data.size() == target.data.size());

        input_ = in;
        target_ = target;

        float sum = 0;

        for (size_t i = 0; i < in.data.size(); i++) {
            float x = in.data[i] - target.data[i];
            sum += x;
        }

        return sum / in.data.size();
    }

    Tensor backward() {
        if (dinput_.dim == 0) {
            dinput_ = input_;
        }

        float div = 1.0 / input_.data.size();

        for (size_t i = 0; i < input_.data.size(); i++) {
            float x = div*(input_.data[i] - target_.data[i]);
            dinput_.data[i] = x;

        }

        return dinput_;
    }

    Tensor target_;
    Tensor input_;
    Tensor dinput_;
};

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "./prog [images-ubyte]\n";
        return 1;
    }

    std::ifstream is(argv[1], std::ios::binary);
    if (!is) {
        std::cerr << "can't open file: " << argv[1] << "\n";
        return 1;
    }

    char magic_str[4];
    char num_images_str[4];
    char num_rows_str[4];
    char num_cols_str[4];
    
    is.read(magic_str, 4);
    is.read(num_images_str, 4);
    is.read(num_rows_str, 4);
    is.read(num_cols_str, 4);

    std::swap(magic_str[0], magic_str[3]);
    std::swap(magic_str[1], magic_str[2]);

    std::swap(num_images_str[0], num_images_str[3]);
    std::swap(num_images_str[1], num_images_str[2]);

    std::swap(num_rows_str[0], num_rows_str[3]);
    std::swap(num_rows_str[1], num_rows_str[2]);

    std::swap(num_cols_str[0], num_cols_str[3]);
    std::swap(num_cols_str[1], num_cols_str[2]);

    int magic = *reinterpret_cast<int*>(magic_str);
    int num_images = *reinterpret_cast<int*>(num_images_str);
    int num_rows = *reinterpret_cast<int*>(num_rows_str);
    int num_cols = *reinterpret_cast<int*>(num_cols_str);

    std::cout << magic << " " << num_images << " " << num_rows << " " << num_cols << "\n";
}

int test(int argc, char **argv) {
    // load MNIST
    // get one image
    Tensor img(1, 3, 3);
    Tensor target(1, 2, 2);
    Conv2D conv1(1, 1, 2, 1);
    MSELoss loss;

    float lr = 0.01;

    for (size_t i = 0; i < img.data.size(); i++) {
        img.data[i] = i;
    }

    conv1.weight_(0, 0, 0, 0) = 1;
    conv1.weight_(0, 0, 0, 1) = 1;
    conv1.weight_(0, 0, 1, 0) = 1;
    conv1.weight_(0, 0, 1, 1) = 1;

    conv1.bias_(0) = -1;

    target(0, 0, 0) = 2;
    target(0, 0, 1) = 3;
    target(0, 1, 0) = 5;
    target(0, 1, 1) = 6;

    Tensor y = conv1(img);
    float l = loss(y, target);

    Tensor delta = loss.backward();
    conv1.backward(delta);


    std::cout << "img: " << img << "\n";
    std::cout << "target: " << target << "\n";
    std::cout << "y: " << y << "\n";
    std::cout << "loss: " << l << "\n";
    std::cout << "delta: " << delta << "\n";
    std::cout << conv1 << "\n";

    conv1.weight_ = conv1.weight_ - conv1.dweight_*lr;
    conv1.bias_ = conv1.bias_ - conv1.bias_*lr;

    y = conv1(img);
    l = loss(y, target);

    std::cout << "y: " << y << "\n";
    std::cout << "loss: " << l << "\n";

    return 0;
}
