#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <immintrin.h>  // AVX2 intrinsics
#include <memory>

class MLP {
private:
    std::vector<int> layer_sizes;
    double learning_rate;
    int num_threads;
    std::mt19937 rng;

    // Flat memory layout for better cache performance
    struct AlignedMatrix {
        double* data;
        size_t rows, cols;
        size_t padded_cols;  // Padded to multiple of 4 for AVX2

        AlignedMatrix(size_t r, size_t c) : rows(r), cols(c) {
            padded_cols = ((c + 3) / 4) * 4;
            size_t total_size = rows * padded_cols;
            data = static_cast<double*>(_mm_malloc(total_size * sizeof(double), 32));
            std::fill(data, data + total_size, 0.0);
        }

        ~AlignedMatrix() {
            if (data) _mm_free(data);
        }

        AlignedMatrix(AlignedMatrix&& other) noexcept
            : data(other.data), rows(other.rows), cols(other.cols), padded_cols(other.padded_cols) {
            other.data = nullptr;
        }

        AlignedMatrix& operator=(AlignedMatrix&& other) noexcept {
            if (this != &other) {
                if (data) _mm_free(data);
                data = other.data;
                rows = other.rows;
                cols = other.cols;
                padded_cols = other.padded_cols;
                other.data = nullptr;
            }
            return *this;
        }

        AlignedMatrix(const AlignedMatrix&) = delete;
        AlignedMatrix& operator=(const AlignedMatrix&) = delete;

        inline double& operator()(size_t r, size_t c) {
            return data[r * padded_cols + c];
        }

        inline const double& operator()(size_t r, size_t c) const {
            return data[r * padded_cols + c];
        }

        inline double* row_ptr(size_t r) {
            return data + r * padded_cols;
        }

        inline const double* row_ptr(size_t r) const {
            return data + r * padded_cols;
        }
    };

    struct AlignedVector {
        double* data;
        size_t size;
        size_t padded_size;

        AlignedVector(size_t s) : size(s) {
            padded_size = ((s + 3) / 4) * 4;
            data = static_cast<double*>(_mm_malloc(padded_size * sizeof(double), 32));
            std::fill(data, data + padded_size, 0.0);
        }

        ~AlignedVector() {
            if (data) _mm_free(data);
        }

        AlignedVector(AlignedVector&& other) noexcept
            : data(other.data), size(other.size), padded_size(other.padded_size) {
            other.data = nullptr;
        }

        AlignedVector& operator=(AlignedVector&& other) noexcept {
            if (this != &other) {
                if (data) _mm_free(data);
                data = other.data;
                size = other.size;
                padded_size = other.padded_size;
                other.data = nullptr;
            }
            return *this;
        }

        AlignedVector(const AlignedVector&) = delete;
        AlignedVector& operator=(const AlignedVector&) = delete;

        inline double& operator[](size_t i) { return data[i]; }
        inline const double& operator[](size_t i) const { return data[i]; }

        void zero() {
            std::fill(data, data + padded_size, 0.0);
        }
    };

    // Network parameters
    std::vector<AlignedMatrix> weights;
    std::vector<AlignedVector> biases;

    // Thread-local storage
    struct ThreadLocalData {
        std::vector<AlignedVector> activations;
        std::vector<AlignedVector> z_values;
        std::vector<AlignedMatrix> weight_gradients;
        std::vector<AlignedVector> bias_gradients;
        std::vector<AlignedVector> deltas;
        double loss;
        int sample_count;

        ThreadLocalData(const std::vector<int>& layer_sizes) : loss(0.0), sample_count(0) {
            for (size_t i = 0; i < layer_sizes.size(); ++i) {
                activations.emplace_back(layer_sizes[i]);
                if (i < layer_sizes.size() - 1) {
                    z_values.emplace_back(layer_sizes[i + 1]);
                    deltas.emplace_back(layer_sizes[i + 1]);
                    bias_gradients.emplace_back(layer_sizes[i + 1]);
                    weight_gradients.emplace_back(layer_sizes[i + 1], layer_sizes[i]);
                }
            }
        }

        void zero_gradients() {
            for (auto& wg : weight_gradients) {
                std::fill(wg.data, wg.data + wg.rows * wg.padded_cols, 0.0);
            }
            for (auto& bg : bias_gradients) {
                bg.zero();
            }
            loss = 0.0;
            sample_count = 0;
        }
    };

    // AVX2 optimized matrix-vector multiplication
    inline void avx2_matrix_vector_multiply(const double* input, const AlignedMatrix& weight_matrix,
                                          double* output, size_t output_size, size_t input_size) {
        const size_t simd_width = 4;

        for (size_t i = 0; i < output_size; ++i) {
            __m256d sum_vec = _mm256_setzero_pd();
            const double* weight_row = weight_matrix.row_ptr(i);
            size_t j = 0;

            // Vectorized loop
            for (; j + simd_width <= input_size; j += simd_width) {
                __m256d input_vec = _mm256_load_pd(&input[j]);
                __m256d weight_vec = _mm256_load_pd(&weight_row[j]);
                sum_vec = _mm256_fmadd_pd(input_vec, weight_vec, sum_vec);
            }

            // Horizontal sum
            double result[4];
            _mm256_store_pd(result, sum_vec);
            double total = result[0] + result[1] + result[2] + result[3];

            // Handle remaining elements
            for (; j < input_size; ++j) {
                total += input[j] * weight_row[j];
            }

            output[i] = total;
        }
    }

    // AVX2 optimized vector addition
    inline void avx2_vector_add(double* vec1, const double* vec2, size_t size) {
        const size_t simd_width = 4;
        size_t i = 0;

        for (; i + simd_width <= size; i += simd_width) {
            __m256d v1 = _mm256_load_pd(&vec1[i]);
            __m256d v2 = _mm256_load_pd(&vec2[i]);
            __m256d result = _mm256_add_pd(v1, v2);
            _mm256_store_pd(&vec1[i], result);
        }

        for (; i < size; ++i) {
            vec1[i] += vec2[i];
        }
    }

    // AVX2 optimized ReLU
    inline void avx2_relu(double* values, size_t size) {
        const size_t simd_width = 4;
        size_t i = 0;
        __m256d zero_vec = _mm256_setzero_pd();

        for (; i + simd_width <= size; i += simd_width) {
            __m256d vals = _mm256_load_pd(&values[i]);
            __m256d result = _mm256_max_pd(vals, zero_vec);
            _mm256_store_pd(&values[i], result);
        }

        for (; i < size; ++i) {
            values[i] = std::max(0.0, values[i]);
        }
    }

    // Standard functions
    double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    void softmax(double* values, size_t size) {
        double max_val = *std::max_element(values, values + size);
        double sum = 0.0;

        for (size_t i = 0; i < size; ++i) {
            values[i] = std::exp(values[i] - max_val);
            sum += values[i];
        }

        if (sum > 0) {
            for (size_t i = 0; i < size; ++i) {
                values[i] /= sum;
            }
        }
    }

public:
    MLP(const std::vector<int>& architecture, double lr, int threads)
        : layer_sizes(architecture), learning_rate(lr), num_threads(threads), rng(std::random_device{}()) {

        // Initialize weights and biases with aligned memory
        for (size_t layer = 0; layer < layer_sizes.size() - 1; ++layer) {
            int fan_in = layer_sizes[layer];
            int fan_out = layer_sizes[layer + 1];
            double std_dev = std::sqrt(2.0 / (fan_in + fan_out));

            weights.emplace_back(fan_out, fan_in);
            biases.emplace_back(fan_out);

            // Xavier initialization
            std::normal_distribution<double> dist(0.0, std_dev);
            for (int i = 0; i < fan_out; ++i) {
                for (int j = 0; j < fan_in; ++j) {
                    weights[layer](i, j) = dist(rng);
                }
                biases[layer][i] = 0.0;
            }
        }
    }

    void forward(const double* input, ThreadLocalData& tld) {
        // Copy input to first activation layer
        std::copy(input, input + layer_sizes[0], tld.activations[0].data);

        // Forward pass through all layers
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            // Matrix-vector multiplication: z = W * a + b
            avx2_matrix_vector_multiply(
                tld.activations[layer].data,
                weights[layer],
                tld.z_values[layer].data,
                layer_sizes[layer + 1],
                layer_sizes[layer]
            );

            // Add bias
            avx2_vector_add(
                tld.z_values[layer].data,
                biases[layer].data,
                layer_sizes[layer + 1]
            );

            // Copy z to activations
            std::copy(tld.z_values[layer].data,
                     tld.z_values[layer].data + layer_sizes[layer + 1],
                     tld.activations[layer + 1].data);

            // Apply activation function
            if (layer < weights.size() - 1) {
                // ReLU for hidden layers
                avx2_relu(tld.activations[layer + 1].data, layer_sizes[layer + 1]);
            } else {
                // Softmax for output layer
                softmax(tld.activations[layer + 1].data, layer_sizes[layer + 1]);
            }
        }
    }

    void backward(const double* target, ThreadLocalData& tld) {
        int num_layers = layer_sizes.size();

        // Calculate output layer deltas (cross-entropy loss with softmax)
        for (size_t i = 0; i < layer_sizes.back(); ++i) {
            tld.deltas[num_layers - 2][i] = tld.activations[num_layers - 1][i] - target[i];
        }

        // Backpropagate deltas through hidden layers
        for (int layer = num_layers - 3; layer >= 0; --layer) {
            for (int neuron = 0; neuron < layer_sizes[layer + 1]; ++neuron) {
                double error = 0.0;

                // Sum weighted errors from next layer
                for (int next_neuron = 0; next_neuron < layer_sizes[layer + 2]; ++next_neuron) {
                    error += tld.deltas[layer + 1][next_neuron] * weights[layer + 1](next_neuron, neuron);
                }

                // Apply ReLU derivative
                tld.deltas[layer][neuron] = error * relu_derivative(tld.z_values[layer][neuron]);
            }
        }

        // Compute gradients
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            // Bias gradients
            for (size_t i = 0; i < layer_sizes[layer + 1]; ++i) {
                tld.bias_gradients[layer][i] += learning_rate * tld.deltas[layer][i];
            }

            // Weight gradients (outer product: delta * activation^T)
            for (size_t i = 0; i < layer_sizes[layer + 1]; ++i) {
                for (size_t j = 0; j < layer_sizes[layer]; ++j) {
                    tld.weight_gradients[layer](i, j) +=
                        learning_rate * tld.deltas[layer][i] * tld.activations[layer][j];
                }
            }
        }
    }

    double calculate_loss(const double* output, const double* target, size_t size) {
        double loss = 0.0;
        for (size_t i = 0; i < size; ++i) {
            if (target[i] > 0) {
                loss -= target[i] * std::log(std::max(output[i], 1e-15));
            }
        }
        return loss;
    }

    int predict(const std::vector<double>& input) {
        ThreadLocalData tld(layer_sizes);
        forward(input.data(), tld);

        double* output = tld.activations.back().data;
        return std::max_element(output, output + layer_sizes.back()) - output;
    }

    void train(const std::vector<std::vector<double>>& train_data,
               const std::vector<int>& train_labels,
               const std::vector<std::vector<double>>& test_data,
               const std::vector<int>& test_labels,
               int epochs, int batch_size = 32) {

        std::vector<int> indices(train_data.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            double total_loss = 0.0;

            for (size_t batch_start = 0; batch_start < train_data.size(); batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, train_data.size());

                // Thread-local storage
                std::vector<ThreadLocalData> thread_data;
                for (int t = 0; t < num_threads; ++t) {
                    thread_data.emplace_back(layer_sizes);
                }

                #pragma omp parallel num_threads(num_threads)
                {
                    int thread_id = omp_get_thread_num();
                    ThreadLocalData& tld = thread_data[thread_id];
                    tld.zero_gradients();

                    std::vector<double> target(layer_sizes.back(), 0.0);

                    #pragma omp for schedule(static)
                    for (size_t i = batch_start; i < batch_end; ++i) {
                        int idx = indices[i];

                        // Create one-hot target
                        std::fill(target.begin(), target.end(), 0.0);
                        target[train_labels[idx]] = 1.0;

                        // Forward pass
                        forward(train_data[idx].data(), tld);

                        // Calculate loss
                        tld.loss += calculate_loss(tld.activations.back().data, target.data(), target.size());
                        tld.sample_count++;

                        // Backward pass
                        backward(target.data(), tld);
                    }
                }

                // Aggregate gradients and update weights
                double batch_loss = 0.0;
                int total_samples = 0;

                for (const auto& tld : thread_data) {
                    batch_loss += tld.loss;
                    total_samples += tld.sample_count;
                }

                // Update parameters
                for (size_t layer = 0; layer < weights.size(); ++layer) {
                    // Update biases
                    for (size_t i = 0; i < layer_sizes[layer + 1]; ++i) {
                        double avg_grad = 0.0;
                        for (const auto& tld : thread_data) {
                            if (tld.sample_count > 0) {
                                avg_grad += tld.bias_gradients[layer][i] / tld.sample_count;
                            }
                        }
                        avg_grad /= num_threads;
                        biases[layer][i] -= avg_grad;
                    }

                    // Update weights
                    for (size_t i = 0; i < layer_sizes[layer + 1]; ++i) {
                        for (size_t j = 0; j < layer_sizes[layer]; ++j) {
                            double avg_grad = 0.0;
                            for (const auto& tld : thread_data) {
                                if (tld.sample_count > 0) {
                                    avg_grad += tld.weight_gradients[layer](i, j) / tld.sample_count;
                                }
                            }
                            avg_grad /= num_threads;
                            weights[layer](i, j) -= avg_grad;
                        }
                    }
                }

                total_loss += batch_loss;
            }

            // Print progress every 10 epochs
//            if ((epoch + 1) % 10 == 0) {
//                int correct = 0;
//                #pragma omp parallel for reduction(+:correct) num_threads(num_threads)
//                for (size_t i = 0; i < test_data.size(); ++i) {
//                    if (predict(test_data[i]) == test_labels[i]) {
//                        correct++;
//                    }
//                }
//                double accuracy = 100.0 * correct / test_data.size();
//                double avg_loss = total_loss / train_data.size();
//                std::cout << "Epoch " << std::setw(3) << epoch + 1
//                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
//                          << " | Test Accuracy: " << std::setprecision(2) << accuracy << "%" << std::endl;
//            }
        }
    }
};

// Utility functions
std::vector<std::vector<double>> load_images(const std::string& filename, int num_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    file.seekg(16);
    std::vector<std::vector<double>> images(num_images, std::vector<double>(784));
    std::vector<unsigned char> raw_data(num_images * 784);
    file.read(reinterpret_cast<char*>(raw_data.data()), num_images * 784);

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < 784; ++j) {
            images[i][j] = raw_data[i * 784 + j] / 255.0;
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

    file.seekg(8);
    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = label;
    }
    return labels;
}

#define NOF_TRAIN 5000
#define NOF_TEST 500
#define LR 0.01
#define EPOCHS 50
#define BATCH_SIZE 32
#define NUM_THREADS 8

int main() {
    std::cout << "Fashion-MNIST MLP Classifier (Fixed AVX2 + Aligned Memory)" << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    std::cout << "\nClass Labels:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << i << ": " << class_names[i] << std::endl;
    }

    std::cout << "Loading training data..." << std::endl;
    auto train_images = load_images("FMNIST/train-images-idx3-ubyte", NOF_TRAIN);
    auto train_labels = load_labels("FMNIST/train-labels-idx1-ubyte", NOF_TRAIN);

    std::cout << "Loading test data..." << std::endl;
    auto test_images = load_images("FMNIST/t10k-images-idx3-ubyte", NOF_TEST);
    auto test_labels = load_labels("FMNIST/t10k-labels-idx1-ubyte", NOF_TEST);

    if (train_images.empty() || test_images.empty()) {
        std::cout << "Failed to load data. Please ensure Fashion-MNIST files are in the current directory." << std::endl;
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
    std::cout << "Using " << NUM_THREADS << " OpenMP threads with AVX2 optimization" << std::endl;

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
Fashion-MNIST MLP Classifier (Fixed AVX2 + Aligned Memory)
===========================================================

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
Using 8 OpenMP threads with AVX2 optimization

Starting training...

Training completed!
Total training time: 30.322 seconds
Average time per epoch: 0.606 seconds

Final Test Accuracy: 85.80%
g++ -fopenmp -mavx2 -mfma -O3 -march=native avx_main.cpp -o avx_main
*/