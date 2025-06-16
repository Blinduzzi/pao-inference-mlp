#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>

class MLP {
private:
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> z_values;
    std::vector<int> layer_sizes;
    double learning_rate;
    std::mt19937 rng;
    int num_threads;

    // Activation functions
    inline double relu(double x) {
        return std::max(0.0, x);
    }

    inline double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-std::max(-500.0, std::min(500.0, x))));
    }

    inline double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    // Softmax for output layer
    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double max_val = *std::max_element(x.begin(), x.end());
        double sum = 0.0;

        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }

        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum;
        }

        return result;
    }

    // Xavier initialization
    void initialize_weights() {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (size_t i = 0; i < weights.size(); ++i) {
            double xavier_scale = std::sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]));

            for (size_t j = 0; j < weights[i].size(); ++j) {
                for (size_t k = 0; k < weights[i][j].size(); ++k) {
                    weights[i][j][k] = dist(rng) * xavier_scale;
                }
                biases[i][j] = dist(rng) * 0.01;
            }
        }
    }

public:
    MLP(const std::vector<int>& sizes, double lr = 0.001, int threads = 8)
        : layer_sizes(sizes), learning_rate(lr), rng(std::random_device{}()), num_threads(threads) {

        // Initialize architecture
        int num_layers = sizes.size();
        weights.resize(num_layers - 1);
        biases.resize(num_layers - 1);
        activations.resize(num_layers);
        z_values.resize(num_layers - 1);

        // Allocate memory for weights and biases
        for (int i = 0; i < num_layers - 1; ++i) {
            weights[i].resize(sizes[i + 1]);
            biases[i].resize(sizes[i + 1]);
            z_values[i].resize(sizes[i + 1]);

            for (int j = 0; j < sizes[i + 1]; ++j) {
                weights[i][j].resize(sizes[i]);
            }
        }

        // Allocate memory for activations
        for (int i = 0; i < num_layers; ++i) {
            activations[i].resize(sizes[i]);
        }

        initialize_weights();
        omp_set_num_threads(num_threads);
    }

    std::vector<double> forward(const std::vector<double>& input, std::vector<std::vector<double>>& local_activations, std::vector<std::vector<double>>& local_z_values) {
        // Set input layer
        local_activations[0] = input;

        // Forward propagation through hidden layers
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
                double sum = biases[layer][neuron];

                for (size_t prev_neuron = 0; prev_neuron < weights[layer][neuron].size(); ++prev_neuron) {
                    sum += weights[layer][neuron][prev_neuron] * local_activations[layer][prev_neuron];
                }

                local_z_values[layer][neuron] = sum;

                // Apply activation function
                if (layer == weights.size() - 1) {
                    // Output layer - softmax
                    local_activations[layer + 1][neuron] = sum;
                } else {
                    // Hidden layers - use ReLU
                    local_activations[layer + 1][neuron] = relu(sum);
                }
            }
        }

        // Apply softmax to output layer
        local_activations.back() = softmax(local_activations.back());
        return local_activations.back();
    }

    void backward(const std::vector<double>& input, const std::vector<double>& target,
                  std::vector<std::vector<double>>& local_activations,
                  std::vector<std::vector<double>>& local_z_values,
                  std::vector<std::vector<std::vector<double>>>& weight_gradients,
                  std::vector<std::vector<double>>& bias_gradients) {
        // Perform forward pass to populate activations
        forward(input, local_activations, local_z_values);

        int num_layers = layer_sizes.size();
        std::vector<std::vector<double>> deltas(num_layers - 1);

        // Initialize deltas
        for (int i = 0; i < num_layers - 1; ++i) {
            deltas[i].resize(layer_sizes[i + 1]);
        }

        // Calculate output layer deltas (cross-entropy loss with softmax)
        for (size_t i = 0; i < local_activations.back().size(); ++i) {
            deltas.back()[i] = local_activations.back()[i] - target[i];
        }

        // Backpropagate deltas through hidden layers
        for (int layer = num_layers - 3; layer >= 0; --layer) {
            for (int neuron = 0; neuron < layer_sizes[layer + 1]; ++neuron) {
                double error = 0.0;

                for (int next_neuron = 0; next_neuron < layer_sizes[layer + 2]; ++next_neuron) {
                    error += deltas[layer + 1][next_neuron] * weights[layer + 1][next_neuron][neuron];
                }

                deltas[layer][neuron] = error * relu_derivative(local_z_values[layer][neuron]);
            }
        }

        // Compute gradients
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
                // Compute bias gradient
                bias_gradients[layer][neuron] = learning_rate * deltas[layer][neuron];

                // Compute weight gradients
                for (size_t prev_neuron = 0; prev_neuron < weights[layer][neuron].size(); ++prev_neuron) {
                    weight_gradients[layer][neuron][prev_neuron] =
                        learning_rate * deltas[layer][neuron] * local_activations[layer][prev_neuron];
                }
            }
        }
    }

    double calculate_loss(const std::vector<double>& output, const std::vector<double>& target) {
        double loss = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            if (target[i] > 0) {
                loss -= target[i] * std::log(std::max(output[i], 1e-15));
            }
        }
        return loss;
    }

    int predict(const std::vector<double>& input) {
        std::vector<std::vector<double>> local_activations = activations;
        std::vector<std::vector<double>> local_z_values = z_values;
        std::vector<double> output = forward(input, local_activations, local_z_values);
        return std::max_element(output.begin(), output.end()) - output.begin();
    }

    void train(const std::vector<std::vector<double>>& train_data,
               const std::vector<int>& train_labels,
               const std::vector<std::vector<double>>& test_data,
               const std::vector<int>& test_labels,
               int epochs, int batch_size = 32) {

        std::vector<int> indices(train_data.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle data for each epoch
            std::shuffle(indices.begin(), indices.end(), rng);
            double total_loss = 0.0;

            // Process data in mini-batches using completely separated thread computation
            for (size_t batch_start = 0; batch_start < train_data.size(); batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, train_data.size());
                size_t actual_batch_size = batch_end - batch_start;

                // Create separate weight and bias update containers for each thread
                std::vector<std::vector<std::vector<std::vector<double>>>> thread_weight_updates(num_threads);
                std::vector<std::vector<std::vector<double>>> thread_bias_updates(num_threads);
                std::vector<double> thread_losses(num_threads, 0.0);
                std::vector<int> thread_sample_counts(num_threads, 0);

                // Initialize each thread's update containers
                for (int t = 0; t < num_threads; ++t) {
                    thread_weight_updates[t] = weights; // Copy structure
                    thread_bias_updates[t] = biases;    // Copy structure

                    // Zero out all updates
                    for (auto& layer : thread_weight_updates[t]) {
                        for (auto& neuron : layer) {
                            std::fill(neuron.begin(), neuron.end(), 0.0);
                        }
                    }
                    for (auto& layer : thread_bias_updates[t]) {
                        std::fill(layer.begin(), layer.end(), 0.0);
                    }
                }

                // Parallel computation with separated threads
                #pragma omp parallel num_threads(num_threads)
                {
                    int thread_id = omp_get_thread_num();

                    // Each thread gets its own copy of network state
                    std::vector<std::vector<double>> local_activations = activations;
                    std::vector<std::vector<double>> local_z_values = z_values;
                    std::vector<std::vector<std::vector<double>>> local_weight_gradients = weights;
                    std::vector<std::vector<double>> local_bias_gradients = biases;

                    // Zero out gradients
                    for (auto& layer : local_weight_gradients) {
                        for (auto& neuron : layer) {
                            std::fill(neuron.begin(), neuron.end(), 0.0);
                        }
                    }
                    for (auto& layer : local_bias_gradients) {
                        std::fill(layer.begin(), layer.end(), 0.0);
                    }

                    // Distribute batch samples among threads
                    #pragma omp for schedule(static)
                    for (size_t i = batch_start; i < batch_end; ++i) {
                        int idx = indices[i];

                        // Create one-hot encoded target
                        std::vector<double> target(10, 0.0);
                        target[train_labels[idx]] = 1.0;

                        // Forward and backward pass
                        std::vector<double> output = forward(train_data[idx], local_activations, local_z_values);
                        thread_losses[thread_id] += calculate_loss(output, target);
                        thread_sample_counts[thread_id]++;

                        backward(train_data[idx], target, local_activations, local_z_values,
                                local_weight_gradients, local_bias_gradients);

                        // Accumulate gradients for this thread
                        for (size_t layer = 0; layer < weights.size(); ++layer) {
                            for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
                                thread_bias_updates[thread_id][layer][neuron] += local_bias_gradients[layer][neuron];
                                for (size_t prev_neuron = 0; prev_neuron < weights[layer][neuron].size(); ++prev_neuron) {
                                    thread_weight_updates[thread_id][layer][neuron][prev_neuron] +=
                                        local_weight_gradients[layer][neuron][prev_neuron];
                                }
                            }
                        }

                        // Reset gradients for next sample
                        for (auto& layer : local_weight_gradients) {
                            for (auto& neuron : layer) {
                                std::fill(neuron.begin(), neuron.end(), 0.0);
                            }
                        }
                        for (auto& layer : local_bias_gradients) {
                            std::fill(layer.begin(), layer.end(), 0.0);
                        }
                    }
                }

                // Average the updates from all threads (master thread combines results)
                // This is inspired from reference [8]
                double batch_loss = 0.0;
                int total_samples = 0;

                for (int t = 0; t < num_threads; ++t) {
                    batch_loss += thread_losses[t];
                    total_samples += thread_sample_counts[t];
                }

                // Apply averaged weight updates
                for (size_t layer = 0; layer < weights.size(); ++layer) {
                    for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
                        // Average bias updates from all threads
                        double avg_bias_update = 0.0;
                        for (int t = 0; t < num_threads; ++t) {
                            if (thread_sample_counts[t] > 0) {
                                avg_bias_update += thread_bias_updates[t][layer][neuron] / thread_sample_counts[t];
                            }
                        }
                        avg_bias_update /= num_threads;
                        biases[layer][neuron] -= avg_bias_update;

                        // Average weight updates from all threads
                        for (size_t prev_neuron = 0; prev_neuron < weights[layer][neuron].size(); ++prev_neuron) {
                            double avg_weight_update = 0.0;
                            for (int t = 0; t < num_threads; ++t) {
                                if (thread_sample_counts[t] > 0) {
                                    avg_weight_update += thread_weight_updates[t][layer][neuron][prev_neuron] / thread_sample_counts[t];
                                }
                            }
                            avg_weight_update /= num_threads;
                            weights[layer][neuron][prev_neuron] -= avg_weight_update;
                        }
                    }
                }

                total_loss += batch_loss;
            }

            // Print progress every 10 epochs
            // if ((epoch + 1) % 10 == 0) {
            //     int correct = 0;
            //     #pragma omp parallel for reduction(+:correct) num_threads(num_threads)
            //     for (size_t i = 0; i < test_data.size(); ++i) {
            //         if (predict(test_data[i]) == test_labels[i]) {
            //             correct++;
            //         }
            //     }
            //     double accuracy = 100.0 * correct / test_data.size();
            //     double avg_loss = total_loss / train_data.size();
            //     std::cout << "Epoch " << std::setw(3) << epoch + 1
            //               << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
            //               << " | Test Accuracy: " << std::setprecision(2) << accuracy << "%" << std::endl;
            // }
        }
    }
};

// Utility functions for loading Fashion-MNIST data
std::vector<std::vector<double>> load_images(const std::string& filename, int num_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    // Skip header (16 bytes for images)
    file.seekg(16);

    std::vector<std::vector<double>> images(num_images, std::vector<double>(784));
    std::vector<unsigned char> raw_data(num_images * 784);

    // Read all pixel data in one go (single-threaded to avoid file I/O race conditions)
    file.read(reinterpret_cast<char*>(raw_data.data()), num_images * 784);

    // Parallelize normalization
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < 784; ++j) {
            images[i][j] = raw_data[i * 784 + j] / 255.0; // Normalize to [0, 1]
        }
    }

    return images;
}

std::vector<int> load_labels(const std::string& filename, int num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    // Skip header (8 bytes for labels)
    file.seekg(8);

    std::vector<int> labels(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = label;
    }

    return labels;
}

# define NOF_TRAIN 5000
# define NOF_TEST 500
# define LR 0.01
# define EPOCHS 50
# define BATCH_SIZE 32
# define NUM_THREADS 8

int main() {
    std::cout << "Fashion-MNIST MLP Classifier (Optimized with Separated Thread OpenMP)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Fashion-MNIST class names
    std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    std::cout << "\nClass Labels:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << i << ": " << class_names[i] << std::endl;
    }

    // Load Fashion-MNIST data
    std::cout << "Loading training data..." << std::endl;
    auto train_images = load_images("FMNIST/train-images-idx3-ubyte", NOF_TRAIN);
    auto train_labels = load_labels("FMNIST/train-labels-idx1-ubyte", NOF_TRAIN);

    std::cout << "Loading test data..." << std::endl;
    auto test_images = load_images("FMNIST/t10k-images-idx3-ubyte", NOF_TEST);
    auto test_labels = load_labels("FMNIST/t10k-labels-idx1-ubyte", NOF_TEST);

    if (train_images.empty() || test_images.empty()) {
        std::cout << "Failed to load data. Please ensure Fashion-MNIST files are in the current directory." << std::endl;
        std::cout << "Download from: https://github.com/zalandoresearch/fashion-mnist" << std::endl;
        return 1;
    }

    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << "Training samples: " << train_images.size() << std::endl;
    std::cout << "Test samples: " << test_images.size() << std::endl;

    std::vector<int> architecture = {784, 128, 64, 10};

    MLP network(architecture, LR, NUM_THREADS);

    std::cout << "\nNetwork Architecture:" << std::endl;
    std::cout << "Input Layer: 784 neurons (28x28 pixels)" << std::endl;
    for (int i = 1; i < architecture.size() - 1; ++i) {
        std::cout << "Hidden Layer " << i << ": " << architecture[i] << " neurons (ReLU)" << std::endl;
    }
    std::cout << "Output Layer: 10 neurons (Softmax)" << std::endl;
    std::cout << "Using " << NUM_THREADS << " OpenMP threads" << std::endl;

    std::cout << "\nStarting training..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    network.train(train_images, train_labels, test_images, test_labels, EPOCHS, BATCH_SIZE);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    double seconds = duration.count() / 1000.0;

    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Total training time: " << std::fixed << std::setprecision(3)
              << seconds << " seconds" << std::endl;
    std::cout << "Average time per epoch: " << std::fixed << std::setprecision(3)
              << seconds / EPOCHS << " seconds" << std::endl;

    // Final evaluation
    int correct = 0;
    #pragma omp parallel for reduction(+:correct) num_threads(NUM_THREADS)
    for (size_t i = 0; i < test_images.size(); ++i) {
        if (network.predict(test_images[i]) == test_labels[i]) {
            correct++;
        }
    }

    double final_accuracy = 100.0 * correct / test_images.size();
    std::cout << "\nFinal Test Accuracy: " << std::fixed << std::setprecision(2)
              << final_accuracy << "%" << std::endl;

    return 0;
}
/*
Fashion-MNIST MLP Classifier (Optimized with Separated Thread OpenMP)
======================================================================

Class Labels:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
Loading training data...
Loading test data...
Data loaded successfully!
Training samples: 5000
Test samples: 500

Network Architecture:
Input Layer: 784 neurons (28x28 pixels)
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 64 neurons (ReLU)
Output Layer: 10 neurons (Softmax)
Using 8 OpenMP threads

Starting training...
Epoch  10 | Loss: 0.5121 | Test Accuracy: 82.40%
Epoch  20 | Loss: 0.4177 | Test Accuracy: 82.80%
Epoch  30 | Loss: 0.3620 | Test Accuracy: 85.40%
Epoch  40 | Loss: 0.3194 | Test Accuracy: 83.40%
Epoch  50 | Loss: 0.2870 | Test Accuracy: 85.60%

Training completed!
Total training time: 86.529 seconds
Average time per epoch: 1.731 seconds

Final Test Accuracy: 85.60%

g++ -fopenmp -O3 openmp_main.cpp -o openmp_main
*/