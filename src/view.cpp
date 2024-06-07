#include <iostream>
#include "mnist.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "./prog [images] [labels] [index]\n";
        return 1;
    }

    MNIST_dataset data(argv[1], argv[2]);

    while (true) {
        int index;
        std::cout << "enter image index: ";
        std::cin >> index;
    
        if (index >= 0 && index < static_cast<int>(data.labels.size())) {
            std::cout << "label: " << data.labels[index] << "\n";
            data.display(index);
        } else {
            std::cout << "out of range\n";
        }
    }

    return 0;
}
