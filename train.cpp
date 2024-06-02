#include "cnn.hpp"
#include <iostream>
#include <algorithm>

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cout << "./prog [train_images] [train_labels] [test_images] [test_labels]\n";
        return 1;
    }

    MNIST mnist(argv[1], argv[2], argv[3], argv[4]);

    // std::vector<Layer*> net {
    //     new Conv2D(1, 4, 2, 2), // --> 4x14x14
    //     new ReLU(),
    //     new Conv2D(4, 8, 2, 2), // --> 8x7x7
    //     new ReLU(),
    //     new Conv2D(8, 8, 2), // --> 8x6x6
    //     new ReLU(),
    //     new Conv2D(8, 10, 2, 2), // --> 10x3x3
    //     new ReLU(),
    //     new Conv2D(10, 10, 3), // --> 10x1x1
    //     new Flatten(),
    //     new Softmax()
    // };

    std::vector<Layer*> net {
        new Conv2D(1, 10, 28), 
        new ReLU(),
        new Flatten(),
        new Softmax()
    };

    CrossEntropyLoss CELoss;
    AccuracyMetric accuracy(10);

    int iterations = 10;
    int batch_size = 64;
    float lr = 0.1;

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
            int k;
            for (k = 0; k < batch_size; k++) {
                if (j + k >= train_idx.size()) {
                     break;
                 }

                int idx = train_idx[j + k];
                Tensor x = mnist.get_train_image(idx);
                int target = mnist.get_train_label(idx);

                for (auto layer: net) {
                    x = (*layer)(x);
                }

                accuracy.update(x, target);

                float loss = CELoss(x, target);
                sum_loss += loss;

                if (!std::isfinite(loss)) {
                    std::cout << "loss: " << loss << "\n";
                    std::cout << x << "\n";
                    std::cout << target << "\n";
                    exit(1);
                }
                Tensor delta = CELoss.backward(); 

                for (int l = net.size() - 1; l >= 0; l--) {
                    delta = net[l]->backward(delta);
                }
            }

            for (auto layer: net) {
                layer->update_weight(lr);
                layer->zero();
            }
        }

        double avg_loss = sum_loss / train_idx.size();
        std::cout << i << ": avg loss: " << avg_loss << " train accuracy: " << accuracy.accuracy() << "\n";
        accuracy.clear();
    }

    return 0;
}

