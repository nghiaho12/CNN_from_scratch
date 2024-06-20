#include <iostream>
#include <algorithm>

#include "cnn.hpp"
#include "mnist.hpp"

// split training index into training + validation
std::tuple<std::vector<size_t>, std::vector<size_t>> split_index(const std::vector<size_t> &train_idx, float validation_per) {
    // you should shuffle train_idx before running this function
   
    assert(validation_per > 0 && validation < 1);

    int valid_size = static_cast<int>(train_idx.size() * validation_per);
    int train_size = static_cast<int>(train_idx.size() - valid_size);

    std::vector<size_t> valid_idx(valid_size);
    std::vector<size_t> new_train_idx(train_size);

    std::copy(train_idx.begin(), train_idx.begin() + train_size, new_train_idx.begin());
    std::copy(train_idx.begin() + train_size, train_idx.end(), valid_idx.begin());

    return {new_train_idx, valid_idx};
}

std::tuple<double, double> mean_stdev(const std::vector<float> &x) {
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double stdev = 0;

    for (float a : x) {
        stdev += (a - mean)*(a - mean);
    }

    stdev = std::sqrt(stdev / x.size());

    return {mean, stdev};
}

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
    float validation_per = 0.05; // percentage of training to use for validation

    // fixed random for testing
    std::default_random_engine random_gen{42};

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

    print_network_info(net);
    init_network_weight(net, random_gen);

    CrossEntropyLoss CELoss;
    AccuracyMetric accuracy(num_classes);

    // create training and validation set
    std::vector<size_t> train_idx(train.labels.size());
    for (size_t k = 0; k < train_idx.size(); k++) {
        train_idx[k] = k;
    } 
    std::ranges::shuffle(train_idx, random_gen);

    std::vector<size_t> valid_idx;
    std::tie(train_idx, valid_idx) = split_index(train_idx, validation_per);

    std::cout << "training size: " << train_idx.size() << "\n";
    std::cout << "validation size: " << valid_idx.size() << "\n";

    // main training loop
    for (int i = 0; i < iterations; i++) {
        std::ranges::shuffle(train_idx, random_gen);
        accuracy.clear();
        double sum_loss = 0;
        std::vector<float> acts[net.size()]; // stats on each layer's output

        for (size_t j = 0; j < train_idx.size(); j+=batch_size) {
            for (int k = 0; k < batch_size; k++) {
                if (j + k >= train_idx.size()) {
                     break;
                 }

                int idx = train_idx[j + k];
                Tensor x = train.get_image(idx);
                int target = train.labels[idx];

                // forward pass
                for (size_t l = 0; l < net.size(); l++) {
                    Layer *layer = net[l];
                    x = (*layer)(x);

                    acts[l].insert(acts[l].end(), x.data.begin(), x.data.end());
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
        float train_accuracy = accuracy.accuracy();

        // accuracy on validation set
        accuracy.clear();
        for (auto idx : valid_idx) {
            Tensor x = train.get_image(idx);
            int target = train.labels[idx];

            for (auto layer: net) {
                x = (*layer)(x);
            }

            accuracy.update(x, target);
        }

        float valid_accuracy = accuracy.accuracy();

        std::cout << "iter " << i << ": avg loss: " << avg_loss << ", train accuracy: " << train_accuracy << ", validation accuracy: " << valid_accuracy << "\n";

        // if stdev is close to zero then the layer's weights are probably in a bad state
        for (size_t j = 0; j < net.size(); j++) {
            auto[mean, stdev] = mean_stdev(acts[j]);
            std::cout << "  layer " << j << ": output mean=" << mean << " stdev=" << stdev << "\n";
        }
    }

    // test dataset
    accuracy.clear();
    for (size_t i = 0; i < test.labels.size(); i++) {
        Tensor x = test.get_image(i);
        for (auto layer: net) {
            x = (*layer)(x);
        }

        accuracy.update(x, test.labels[i]);
    }

    std::cout << "\n";
    std::cout << "test accuracy: " << accuracy.accuracy() << "\n";
    std::cout << "confusion matrix for test set\n";
    accuracy.print_confusion_matrix();

    // clean up memory
    for (auto layer : net) {
        delete layer;
    }

    return 0;
}

