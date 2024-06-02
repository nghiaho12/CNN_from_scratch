#include "cnn.hpp"
#include <iostream>

const float TOL = 0.01;
const float DELTA = 0.001;

void test_ReLU() {
    Tensor x(4);
    Tensor delta(4);
    ReLU relu;

    x(0) = -1;
    x(1) = -2;
    x(2) = 3;
    x(3) = 4;

    delta(0) = 2;
    delta(1) = 2;
    delta(2) = 2;
    delta(3) = 2;

    Tensor y = relu(x);
    Tensor deriv = relu.backward(delta);

    assert(y(0) == 0);
    assert(y(1) == 0);
    assert(y(2) == 3);
    assert(y(3) == 4);

    assert(deriv(0) == 0);
    assert(deriv(1) == 0);
    assert(deriv(2) == 2);
    assert(deriv(3) == 2);
}

void test_Conv2D() {
    Tensor img(1, 28, 28);

    int in_channels = 1;
    int out_channels = 4;
    int ksize = 2;
    int stride = 2;

    Conv2D conv(in_channels, out_channels, ksize, stride);

    img.set_random();

    Tensor y = conv(img);
    assert(y.shape(0) == 4);
    assert(y.shape(1) == 14);
    assert(y.shape(2) == 14);

    // check derivative
    Tensor delta = y;
    delta.set_one();

    Tensor deriv = conv.backward(delta);
   
    for (int i = 0; i < img.shape(1); i++) {
        for (int j = 0; j < img.shape(2); j++) {
            Tensor img2 = img;
            img2(0, i, j) += DELTA;
            Tensor y2 = conv(img2);

            float d = (y2 - y).sum() / DELTA;
            assert(std::abs(deriv(0, i, j) - d) < TOL);
        }
    }
}

void test_Softmax() {
    Tensor x(3);
    Tensor delta(3);
    Softmax softmax;

    x(0) = -1;
    x(1) = 0; 
    x(2) = 1;

    delta(0) = 1;
    delta(1) = 0;
    delta(2) = 0;

    Tensor y = softmax(x);
    Tensor deriv = softmax.backward(delta);

    assert(std::abs(y(0) - 0.090031) < TOL);
    assert(std::abs(y(1) - 0.244728) < TOL);
    assert(std::abs(y(2) - 0.665241) < TOL);

    // numerical derivative
    x(0) += DELTA;
    Tensor y1 = softmax(x);

    for (int i = 0; i < x.shape(0); i++) {
        float d = (y1(i) - y(i)) / DELTA;
        assert(std::abs(deriv(i) - d) < TOL);
    }
}

void test_CrossEntropyLoss() {
    Tensor y(3);
    CrossEntropyLoss CELoss;
    int target = 1;

    y(0) = 0.25;
    y(1) = 0.50;
    y(2) = 0.25;

    float loss = CELoss(y, target);
    assert(std::abs(loss - 1.2685) < TOL);

    Tensor deriv = CELoss.backward();
    assert(std::abs(deriv(0) - 1.3333) < TOL);
    assert(std::abs(deriv(1) - -2) < TOL);
    assert(std::abs(deriv(2) - 1.3333) < TOL);

    // numerical derivative
    for (int i = 0; i < y.shape(0); i++) {
        Tensor y2 = y;
        y2(i) += DELTA;
        float loss2 = CELoss(y2, target);
        float dloss = (loss2 - loss) / DELTA;
        assert(std::abs(deriv(i) - dloss) < TOL);
    }   
}

void test_Flatten() {    
    Tensor x(1, 2, 2);
    Tensor delta(4);
    Flatten flatten;

    x(0,0,0) = 1;
    x(0,0,1) = 2;
    x(0,1,0) = 3;
    x(0,1,1) = 4;

    delta(0) = -1;
    delta(1) = -2;
    delta(2) = -3;
    delta(3) = -4;

    Tensor y = flatten(x);

    assert(y.shape().size() == 1);
    assert(y(0) == 1);
    assert(y(1) == 2);
    assert(y(2) == 3);
    assert(y(3) == 4);

    Tensor d = flatten.backward(delta);

    assert(d.shape().size() == 3);
    assert(d(0,0,0) == -1);
    assert(d(0,0,1) == -2);
    assert(d(0,1,0) == -3);
    assert(d(0,1,1) == -4);
}

int main(int argc, char **argv) {
    test_ReLU();
    test_Conv2D();
    test_Softmax();
    test_CrossEntropyLoss();
    test_Flatten();

    std::cout << "All tests passed\n";
}

