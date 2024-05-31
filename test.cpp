#include "cnn.hpp"
#include <iostream>

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
    Tensor img2(1, 28, 28);

    int in_channels = 1;
    int out_channels = 4;
    int ksize = 2;
    int stride = 2;

    Conv2D conv1(in_channels, out_channels, ksize, stride);

    img.set_random();

    Tensor y = conv1(img);
    assert(y.shape(0) == 4);
    assert(y.shape(1) == 14);
    assert(y.shape(2) == 14);

    // check derivative
    Tensor delta = y;
    delta.set_one();

    Tensor dinput = conv1.backward(delta);

    float eps = 0.0001;
    float tol = 0.001;

    img2 = img;
    img2(0, 14, 14) += eps;
    Tensor y2 = conv1(img2);

    float numeric_dinput = 0;
    for (int i = 0; i < y.shape(0); i++) {
        for (int j = 0; j < y.shape(1); j++) {}
        numeric_dinput += (y2(i, 7, 7) - y(i, 7, 7)) / eps;
    }

    std::cout << dinput(0, 14, 14) << " == " << numeric_dinput << "\n";
    assert(std::abs(dinput(0, 14, 14) - numeric_dinput) < tol);
}

void test_Softmax() {
    Tensor x(3);
    Tensor delta(3);
    Softmax softmax;

    constexpr float tol = 0.001;

    x(0) = -1;
    x(1) = 0;
    x(2) = 1;

    Tensor y = softmax(x);

    assert(std::abs(y(0) - 0.090031) < tol);
    assert(std::abs(y(1) - 0.244728) < tol);
    assert(std::abs(y(2) - 0.665241) < tol);

    delta(0) = 0;
    delta(1) = 0;
    delta(2) = 1;
  
    Tensor deriv = softmax.backward(delta);
    assert(std::abs(deriv(0) - -0.665241*0.090031) < tol);
    assert(std::abs(deriv(1) - -0.665241*0.244728) < tol);
    assert(std::abs(deriv(2) - 0.665241*(1 - 0.665241)) < tol);
}

int main(int argc, char **argv) {
    test_ReLU();
    test_Conv2D();
    test_Softmax();

    std::cout << "All tests passed\n";
}

