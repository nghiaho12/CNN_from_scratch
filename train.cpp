#include "cnn.hpp"
#include <iostream>
#include <algorithm>

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cout << "./prog [train_images] [train_labels] [test_images] [test_labels]\n";
        return 1;
    }

    MNIST mnist(argv[1], argv[2], argv[3], argv[4]);

    std::vector<Layer> net {
        Conv2D(1, 4, 2, 2), // --> 4x14x14
        ReLU(),
        Conv2D(4, 8, 2, 2), // --> 8x7x7
        ReLU(),
        Conv2D(8, 8, 2), // --> 8x6x6
        ReLU(),
        Conv2D(8, 10, 2, 2), // --> 10x3x3
        ReLU(),
        Conv2D(10, 10, 3), // --> 10x1x1
        ReLU(),
        Flatten()
    };

    CrossEntropyLoss CELoss;

    int iterations = 10;
    int batch_size = 128;
    float lr = 0.01;

    std::random_device rd;
    std::mt19937 gen {rd()};

    // index to training images
    std::vector<size_t> train_idx(mnist.train_.shape(0));
    for (size_t k = 0; k < train_idx.size(); k++) {
        train_idx[k] = k;
    } 

    for (int i = 0; i < iterations; i++) {
        std::ranges::shuffle(train_idx, gen);
        double sum_loss = 0;

        for (size_t j = 0; j < train_idx.size(); j+=batch_size) {
            // zzero out gradients
            for (auto& layer: net) {
                layer.zero();
            }

            for (int k = 0; k < batch_size; k++) {
                if (j + k >= train_idx.size()) {
                     break;
                 }

                int idx = train_idx[j + k];
                Tensor x = mnist.get_train_image(idx);
                int target = mnist.get_train_label(idx);

                // forward pass
                for (auto& layer: net) {
                    x = layer(x); 
                }

                float loss = CELoss(x, target);
                sum_loss += loss;

                Tensor delta = CELoss.backward(); 

                // backward pass
                for (int l = net.size() - 1; l >= 0; l--) {
                     delta = net[l].backward(delta);
                }
            }
        }

        double avg_loss = sum_loss / train_idx.size();
        std::cout << i << ": avg loss: " << avg_loss << "\n";
    }

    return 0;
}

