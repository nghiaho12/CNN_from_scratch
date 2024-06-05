#include "cnn.hpp"
#include <iostream>
#include <algorithm>

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cout << "./prog [train_images] [train_labels] [test_images] [test_labels]\n";
        return 1;
    }

    int iterations = 10;
    int batch_size = 64;
    float lr = 0.01;
    float momentum = 0.9;
    int num_classes = 10;
    std::default_random_engine random_gen{42};

    CrossEntropyLoss CELoss;
    AccuracyMetric accuracy(num_classes);

    MNIST_dataset train(argv[1], argv[2]);
    MNIST_dataset test(argv[3], argv[4]);

    // std::vector<Layer*> net {
    //     new Flatten(),
    //     new Dense(28*28, 32),
    //     new ReLU(),
    //     new Dense(32, 10),
    //     new Softmax()
    // };

    std::vector<Layer*> net {
        new Conv2D(1, 4, 2, 2), // --> 4x14x14
        new ReLU(),
        new Conv2D(4, 8, 2, 2), // --> 8x7x7
        new ReLU(),
        new Conv2D(8, 8, 2, 1), // --> 8x6x6
        new ReLU(),
        new Conv2D(8, 16, 2, 2), // --> 16x3x3
        new ReLU(),
        new Conv2D(16, 10, 3), // --> 10x1x1
        new Flatten(),
        new Softmax()
    };

    init_weight_kaiming_he(net, random_gen);

    // print some info about the network
    int total_params = 0;
    for (auto &l : net) {
        std::cout << *l << "\n";
        total_params += l->weight.data.size() + l->bias.data.size();
    }
    std::cout << "\ntotal trainable params: " << total_params << "\n";
    
    // index to training images
    std::vector<size_t> train_idx(train.labels.size());
    for (size_t k = 0; k < train_idx.size(); k++) {
        train_idx[k] = k;
    } 

    // main training loop
    for (int i = 0; i < iterations; i++) {
        std::ranges::shuffle(train_idx, random_gen);
        accuracy.clear();
        double sum_loss = 0;

        for (size_t j = 0; j < train_idx.size(); j+=batch_size) {
            for (int k = 0; k < batch_size; k++) {
                if (j + k >= train_idx.size()) {
                     break;
                 }

                int idx = train_idx[j + k];
                Tensor x = train.get_image(idx);
                int target = train.labels[idx];

                // forward pass
                for (Layer* layer: net) {
                    x = (*layer)(x);
                }

                accuracy.update(x, target);

                float loss = CELoss(x, target);
                sum_loss += loss;

                // backward pass
                Tensor delta = CELoss.backward(); 
                for (int l = net.size() - 1; l >= 0; l--) {
                    delta = net[l]->backward(delta);
                }
            }

            SGD_weight_update(net, lr, momentum);
        }

        double avg_loss = sum_loss / train_idx.size();
        std::cout << i << ": avg loss: " << avg_loss << " train accuracy: " << accuracy.accuracy() << "\n";
    }

    // test dataset
    accuracy.clear();
    for (size_t i = 0; i < test.labels.size(); i++) {
        Tensor x = test.get_image(i);
        for (Layer* layer: net) {
            x = (*layer)(x);
        }

        accuracy.update(x, test.labels[i]);
    }

    std::cout << "\n";
    std::cout << "test accuracy: " << accuracy.accuracy() << "\n";
    std::cout << "confusion matrix: " << accuracy.confusion_matrix() << "\n";

    // clean up memory
    for (auto l : net) {
        delete l;
    }

    return 0;
}

