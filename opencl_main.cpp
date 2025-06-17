#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <cstring>

// Define OpenCL version to avoid warnings
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl.hpp>

using namespace std::chrono;
using namespace std;

// Configuration constants
#define NOF_TRAIN 5000
#define NOF_TEST 500
#define LR 0.01f
#define EPOCHS 50
#define BATCH_SIZE 64  // Larger batch for GPU efficiency

// Helper function to check OpenCL errors
#define cl_check(result) { if (result != CL_SUCCESS) { \
    cout << "OpenCL error at line " << __LINE__ << \
    " with code " << result << std::endl; \
    exit(result); } \
}

// Simplified aligned memory allocation that works across platforms
template<typename T>
T* aligned_alloc(size_t count, size_t alignment = 32) {
    size_t size = count * sizeof(T);

    // Use regular malloc for simplicity - modern systems handle this well
    void* ptr = malloc(size);
    if (!ptr) {
        throw std::bad_alloc();
    }

    // Clear the memory
    memset(ptr, 0, size);
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free(T* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// Optimized MLP class using OpenCL
class OpenCL_MLP {
private:
    vector<int> layer_sizes;
    float learning_rate;
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    cl::Program program;

    // OpenCL kernels
    cl::Kernel forward_kernel;
    cl::Kernel relu_kernel;
    cl::Kernel softmax_kernel;
    cl::Kernel backward_kernel;
    cl::Kernel update_weights_kernel;

    // Device buffers for weights and biases
    vector<cl::Buffer> weight_buffers;
    vector<cl::Buffer> bias_buffers;
    vector<cl::Buffer> activation_buffers;
    vector<cl::Buffer> gradient_buffers;
    vector<cl::Buffer> delta_buffers;

    // Host memory (aligned)
    vector<float*> weights;
    vector<float*> biases;
    vector<int> weight_sizes;
    vector<int> bias_sizes;

    mt19937 rng;

public:
    OpenCL_MLP(const vector<int>& architecture, float lr)
        : layer_sizes(architecture), learning_rate(lr), rng(42) {

        initializeOpenCL();
        initializeNetwork();
        createKernels();
        allocateBuffers();
    }

    ~OpenCL_MLP() {
        // Free aligned memory
        for (auto* ptr : weights) aligned_free(ptr);
        for (auto* ptr : biases) aligned_free(ptr);
    }

private:
    void initializeOpenCL() {
        // Get OpenCL platforms
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw runtime_error("No OpenCL platforms found!");
        }

        // Find GPU device (prefer discrete GPU)
        cl::Platform selectedPlatform;
        bool foundGPU = false;

        for (auto& platform : platforms) {
            vector<cl::Device> devices;
            try {
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                if (!devices.empty()) {
                    device = devices[0];
                    selectedPlatform = platform;
                    foundGPU = true;
                    break;
                }
            } catch (...) {
                continue;
            }
        }

        if (!foundGPU) {
            cout << "No GPU found, using CPU..." << endl;
            vector<cl::Device> cpuDevices;
            platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &cpuDevices);
            if (!cpuDevices.empty()) {
                device = cpuDevices[0];
                selectedPlatform = platforms[0];
            } else {
                throw runtime_error("No OpenCL devices found!");
            }
        }

        cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << endl;
        cout << "Max compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
        cout << "Max work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;

        // Create context and command queue
        cl_int err;
        context = cl::Context({device}, nullptr, nullptr, nullptr, &err);
        cl_check(err);

        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        cl_check(err);
    }

    void initializeNetwork() {
        // Calculate sizes and allocate aligned memory
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            int weight_count = layer_sizes[i] * layer_sizes[i + 1];
            int bias_count = layer_sizes[i + 1];

            weight_sizes.push_back(weight_count);
            bias_sizes.push_back(bias_count);

            // Allocate aligned memory
            float* weight_data = aligned_alloc<float>(weight_count);
            float* bias_data = aligned_alloc<float>(bias_count);

            weights.push_back(weight_data);
            biases.push_back(bias_data);

            // Xavier initialization
            float scale = sqrt(2.0f / (layer_sizes[i] + layer_sizes[i + 1]));
            normal_distribution<float> dist(0.0f, scale);

            for (int j = 0; j < weight_count; j++) {
                weight_data[j] = dist(rng);
            }

            fill_n(bias_data, bias_count, 0.0f);
        }
    }

    void createKernels() {
        string kernelSource = R"CLC(
// Optimized matrix-vector multiplication for neural networks
__kernel void matrix_vector_multiply(
    __global const float* weights,
    __global const float* input,
    __global const float* biases,
    __global float* output,
    const int input_size,
    const int output_size,
    const int batch_size
) {
    const int batch_idx = get_global_id(0);
    const int output_idx = get_global_id(1);

    if (batch_idx >= batch_size || output_idx >= output_size) return;

    float sum = biases[output_idx];

    // Compute dot product for this output neuron
    for (int i = 0; i < input_size; i++) {
        sum += input[batch_idx * input_size + i] *
               weights[output_idx * input_size + i];
    }

    output[batch_idx * output_size + output_idx] = sum;
}

// ReLU activation function
__kernel void relu_activation(
    __global float* data,
    const int size
) {
    const int gid = get_global_id(0);
    if (gid >= size) return;

    data[gid] = max(data[gid], 0.0f);
}

// Softmax activation function
__kernel void softmax_activation(
    __global float* data,
    __local float* local_max,
    __local float* local_sum,
    const int batch_size,
    const int class_size
) {
    const int batch_idx = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    if (batch_idx >= batch_size) return;

    __global float* batch_data = data + batch_idx * class_size;

    // Find maximum value for numerical stability
    float max_val = -FLT_MAX;
    for (int i = lid; i < class_size; i += local_size) {
        max_val = max(max_val, batch_data[i]);
    }

    local_max[lid] = max_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction to find global maximum
    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            local_max[lid] = max(local_max[lid], local_max[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float global_max = local_max[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = lid; i < class_size; i += local_size) {
        float exp_val = exp(batch_data[i] - global_max);
        batch_data[i] = exp_val;
        sum += exp_val;
    }

    local_sum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction to find total sum
    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float total_sum = local_sum[0];

    // Normalize
    for (int i = lid; i < class_size; i += local_size) {
        batch_data[i] /= total_sum;
    }
}

// Backward pass kernel for computing deltas
__kernel void compute_deltas(
    __global const float* activations,
    __global const float* targets,
    __global const float* weights,
    __global const float* next_deltas,
    __global float* deltas,
    const int batch_size,
    const int layer_size,
    const int next_layer_size,
    const int is_output_layer
) {
    const int batch_idx = get_global_id(0);
    const int neuron_idx = get_global_id(1);

    if (batch_idx >= batch_size || neuron_idx >= layer_size) return;

    int idx = batch_idx * layer_size + neuron_idx;
    float activation = activations[idx];
    float delta = 0.0f;

    if (is_output_layer) {
        // Output layer: compute softmax derivative
        float target = targets[batch_idx * layer_size + neuron_idx];
        delta = activation - target;
    } else {
        // Hidden layer: compute derivative using next layer
        for (int i = 0; i < next_layer_size; i++) {
            delta += next_deltas[batch_idx * next_layer_size + i] *
                     weights[i * layer_size + neuron_idx];
        }
        // ReLU derivative
        if (activation <= 0.0f) delta = 0.0f;
    }

    deltas[idx] = delta;
}

// Update weights kernel
__kernel void update_weights(
    __global float* weights,
    __global float* biases,
    __global const float* activations,
    __global const float* deltas,
    const float learning_rate,
    const int batch_size,
    const int input_size,
    const int output_size
) {
    const int output_idx = get_global_id(0);
    const int input_idx = get_global_id(1);

    if (output_idx >= output_size) return;

    if (input_idx < input_size) {
        // Update weights
        float gradient = 0.0f;
        for (int batch = 0; batch < batch_size; batch++) {
            gradient += deltas[batch * output_size + output_idx] *
                       activations[batch * input_size + input_idx];
        }
        gradient /= batch_size;

        int weight_idx = output_idx * input_size + input_idx;
        weights[weight_idx] -= learning_rate * gradient;
    }

    // Update biases (only one thread per output neuron)
    if (input_idx == 0) {
        float bias_gradient = 0.0f;
        for (int batch = 0; batch < batch_size; batch++) {
            bias_gradient += deltas[batch * output_size + output_idx];
        }
        bias_gradient /= batch_size;

        biases[output_idx] -= learning_rate * bias_gradient;
    }
}

// Simple forward pass kernel
__kernel void forward_pass(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int batch_size,
    const int input_size,
    const int output_size
) {
    const int gid = get_global_id(0);
    const int batch_idx = gid / output_size;
    const int neuron_idx = gid % output_size;

    if (batch_idx >= batch_size || neuron_idx >= output_size) return;

    float sum = biases[neuron_idx];

    for (int i = 0; i < input_size; i++) {
        sum += input[batch_idx * input_size + i] *
               weights[neuron_idx * input_size + i];
    }

    output[batch_idx * output_size + neuron_idx] = sum;
}
)CLC";

        cl_int err;
        cl::Program::Sources sources;
        sources.push_back({kernelSource.c_str(), kernelSource.length()});

        program = cl::Program(context, sources, &err);
        cl_check(err);

        string buildOptions = "-cl-mad-enable -cl-fast-relaxed-math -cl-unsafe-math-optimizations";
        err = program.build({device}, buildOptions.c_str());
        if (err != CL_SUCCESS) {
            cout << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
            cl_check(err);
        }

        // Create kernels with correct names
        forward_kernel = cl::Kernel(program, "forward_pass", &err);
        cl_check(err);

        relu_kernel = cl::Kernel(program, "relu_activation", &err);
        cl_check(err);

        softmax_kernel = cl::Kernel(program, "softmax_activation", &err);
        cl_check(err);

        backward_kernel = cl::Kernel(program, "compute_deltas", &err);
        cl_check(err);

        update_weights_kernel = cl::Kernel(program, "update_weights", &err);
        cl_check(err);

        cout << "OpenCL kernels compiled successfully!" << endl;
    }

    void allocateBuffers() {
        cl_int err;

        // Allocate weight and bias buffers
        for (size_t i = 0; i < weights.size(); i++) {
            cl::Buffer weight_buf(context, CL_MEM_READ_WRITE,
                                weight_sizes[i] * sizeof(float), nullptr, &err);
            cl_check(err);
            weight_buffers.push_back(weight_buf);

            cl::Buffer bias_buf(context, CL_MEM_READ_WRITE,
                              bias_sizes[i] * sizeof(float), nullptr, &err);
            cl_check(err);
            bias_buffers.push_back(bias_buf);
        }

        // Allocate activation, gradient and delta buffers
        for (int layer_size : layer_sizes) {
            cl::Buffer activation_buf(context, CL_MEM_READ_WRITE,
                                    layer_size * BATCH_SIZE * sizeof(float), nullptr, &err);
            cl_check(err);
            activation_buffers.push_back(activation_buf);

            cl::Buffer gradient_buf(context, CL_MEM_READ_WRITE,
                                  layer_size * BATCH_SIZE * sizeof(float), nullptr, &err);
            cl_check(err);
            gradient_buffers.push_back(gradient_buf);

            cl::Buffer delta_buf(context, CL_MEM_READ_WRITE,
                               layer_size * BATCH_SIZE * sizeof(float), nullptr, &err);
            cl_check(err);
            delta_buffers.push_back(delta_buf);
        }

        // Copy initial weights to device
        for (size_t i = 0; i < weights.size(); i++) {
            err = queue.enqueueWriteBuffer(weight_buffers[i], CL_TRUE, 0,
                                         weight_sizes[i] * sizeof(float), weights[i]);
            cl_check(err);

            err = queue.enqueueWriteBuffer(bias_buffers[i], CL_TRUE, 0,
                                         bias_sizes[i] * sizeof(float), biases[i]);
            cl_check(err);
        }
    }

public:
    void train(const vector<vector<float>>& train_data,
               const vector<int>& train_labels,
               const vector<vector<float>>& test_data,
               const vector<int>& test_labels,
               int epochs) {

        cout << "Starting OpenCL training with optimized kernels..." << endl;

        vector<int> indices(train_data.size());
        iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            shuffle(indices.begin(), indices.end(), rng);

            auto epoch_start = high_resolution_clock::now();
            double total_loss = 0.0;
            int batches_processed = 0;

            // Process in batches
            for (size_t batch_start = 0; batch_start < train_data.size(); batch_start += BATCH_SIZE) {
                size_t batch_end = min(batch_start + BATCH_SIZE, train_data.size());
                size_t current_batch_size = batch_end - batch_start;

                // Prepare batch data
                vector<float> batch_inputs(current_batch_size * layer_sizes[0]);
                vector<int> batch_labels(current_batch_size);

                for (size_t i = 0; i < current_batch_size; i++) {
                    int idx = indices[batch_start + i];
                    copy(train_data[idx].begin(), train_data[idx].end(),
                         batch_inputs.begin() + i * layer_sizes[0]);
                    batch_labels[i] = train_labels[idx];
                }

                // Forward and backward pass on GPU
                float batch_loss = trainBatch(batch_inputs, batch_labels, current_batch_size);
                total_loss += batch_loss;
                batches_processed++;
            }

            auto epoch_end = high_resolution_clock::now();
            auto epoch_time = duration_cast<milliseconds>(epoch_end - epoch_start);

            if (epoch % 5 == 0) {
                double accuracy = evaluate(test_data, test_labels);
                cout << "Epoch " << epoch + 1 << "/" << epochs
                     << " - Loss: " << fixed << setprecision(4) << total_loss / batches_processed
                     << " - Accuracy: " << fixed << setprecision(2) << accuracy << "%"
                     << " - Time: " << epoch_time.count() << "ms" << endl;
            }
        }
    }

private:
    float trainBatch(const vector<float>& batch_inputs, const vector<int>& batch_labels, size_t batch_size) {
        cl_int err;

        // Upload batch to GPU
        err = queue.enqueueWriteBuffer(activation_buffers[0], CL_TRUE, 0,
                                     batch_size * layer_sizes[0] * sizeof(float),
                                     batch_inputs.data());
        cl_check(err);

        // Forward pass through all layers
        for (size_t layer = 0; layer < layer_sizes.size() - 1; layer++) {
            // Set kernel arguments for forward pass
            forward_kernel.setArg(0, activation_buffers[layer]);      // input
            forward_kernel.setArg(1, weight_buffers[layer]);          // weights
            forward_kernel.setArg(2, bias_buffers[layer]);            // biases
            forward_kernel.setArg(3, activation_buffers[layer + 1]);  // output
            forward_kernel.setArg(4, (int)batch_size);
            forward_kernel.setArg(5, layer_sizes[layer]);             // input_size
            forward_kernel.setArg(6, layer_sizes[layer + 1]);         // output_size

            // Execute forward pass
            size_t global_work_size = batch_size * layer_sizes[layer + 1];
            err = queue.enqueueNDRangeKernel(forward_kernel, cl::NullRange,
                                           cl::NDRange(global_work_size), cl::NullRange);
            cl_check(err);

            // Apply activation function
            if (layer < layer_sizes.size() - 2) {
                // ReLU for hidden layers
                size_t activation_size = batch_size * layer_sizes[layer + 1];
                relu_kernel.setArg(0, activation_buffers[layer + 1]);
                relu_kernel.setArg(1, (int)activation_size);

                err = queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange,
                                               cl::NDRange(activation_size), cl::NullRange);
                cl_check(err);
            } else {
                // Softmax for output layer
                // FIX: Cast to size_t to avoid type mismatch
                size_t max_work_group_size = static_cast<size_t>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
                size_t local_work_size = min(static_cast<size_t>(64), max_work_group_size);

                softmax_kernel.setArg(0, activation_buffers[layer + 1]);
                softmax_kernel.setArg(1, cl::Local(local_work_size * sizeof(float))); // local_max
                softmax_kernel.setArg(2, cl::Local(local_work_size * sizeof(float))); // local_sum
                softmax_kernel.setArg(3, (int)batch_size);
                softmax_kernel.setArg(4, layer_sizes[layer + 1]);

                err = queue.enqueueNDRangeKernel(softmax_kernel, cl::NullRange,
                                               cl::NDRange(batch_size * local_work_size),
                                               cl::NDRange(local_work_size));
                cl_check(err);
            }
        }

        // Create one-hot encoded targets
        vector<float> targets(batch_size * layer_sizes.back(), 0.0f);
        for (size_t i = 0; i < batch_size; i++) {
            targets[i * layer_sizes.back() + batch_labels[i]] = 1.0f;
        }

        // Upload targets to GPU
        cl::Buffer target_buffer(context, CL_MEM_READ_ONLY,
                               targets.size() * sizeof(float), nullptr, &err);
        cl_check(err);
        err = queue.enqueueWriteBuffer(target_buffer, CL_TRUE, 0,
                                     targets.size() * sizeof(float), targets.data());
        cl_check(err);

        // Backward pass
        for (int layer = layer_sizes.size() - 2; layer >= 0; layer--) {
            int is_output_layer = (layer == layer_sizes.size() - 2) ? 1 : 0;
            int next_layer_size = (layer < layer_sizes.size() - 2) ? layer_sizes[layer + 2] : 0;

            // Compute deltas
            backward_kernel.setArg(0, activation_buffers[layer + 1]);
            backward_kernel.setArg(1, target_buffer);
            backward_kernel.setArg(2, (layer < layer_sizes.size() - 2) ? weight_buffers[layer + 1] : weight_buffers[layer]);
            backward_kernel.setArg(3, (layer < layer_sizes.size() - 2) ? delta_buffers[layer + 2] : delta_buffers[layer + 1]);
            backward_kernel.setArg(4, delta_buffers[layer + 1]);
            backward_kernel.setArg(5, (int)batch_size);
            backward_kernel.setArg(6, layer_sizes[layer + 1]);
            backward_kernel.setArg(7, next_layer_size);
            backward_kernel.setArg(8, is_output_layer);

            size_t global_work_size[2] = {batch_size, static_cast<size_t>(layer_sizes[layer + 1])};
            err = queue.enqueueNDRangeKernel(backward_kernel, cl::NullRange,
                                           cl::NDRange(global_work_size[0], global_work_size[1]), cl::NullRange);
            cl_check(err);

            // Update weights
            update_weights_kernel.setArg(0, weight_buffers[layer]);
            update_weights_kernel.setArg(1, bias_buffers[layer]);
            update_weights_kernel.setArg(2, activation_buffers[layer]);
            update_weights_kernel.setArg(3, delta_buffers[layer + 1]);
            update_weights_kernel.setArg(4, learning_rate);
            update_weights_kernel.setArg(5, (int)batch_size);
            update_weights_kernel.setArg(6, layer_sizes[layer]);
            update_weights_kernel.setArg(7, layer_sizes[layer + 1]);

            size_t weight_global_size[2] = {static_cast<size_t>(layer_sizes[layer + 1]),
                                          static_cast<size_t>(layer_sizes[layer])};
            err = queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange,
                                           cl::NDRange(weight_global_size[0], weight_global_size[1]), cl::NullRange);
            cl_check(err);
        }

        queue.finish();

        // Calculate simple cross-entropy loss
        vector<float> output_data(batch_size * layer_sizes.back());
        err = queue.enqueueReadBuffer(activation_buffers.back(), CL_TRUE, 0,
                                    output_data.size() * sizeof(float), output_data.data());
        cl_check(err);

        float loss = 0.0f;
        for (size_t i = 0; i < batch_size; i++) {
            float prob = output_data[i * layer_sizes.back() + batch_labels[i]];
            loss -= log(max(prob, 1e-15f));  // Avoid log(0)
        }

        return loss / batch_size;
    }

    double evaluate(const vector<vector<float>>& test_data, const vector<int>& test_labels) {
        int correct = 0;

        // Process test data in smaller batches
        for (size_t i = 0; i < test_data.size(); i += BATCH_SIZE) {
            size_t batch_end = min(i + BATCH_SIZE, test_data.size());
            size_t batch_size = batch_end - i;

            vector<float> batch_inputs(batch_size * layer_sizes[0]);
            for (size_t j = 0; j < batch_size; j++) {
                copy(test_data[i + j].begin(), test_data[i + j].end(),
                     batch_inputs.begin() + j * layer_sizes[0]);
            }

            // Forward pass
            vector<float> predictions = predict(batch_inputs, batch_size);

            // Count correct predictions
            for (size_t j = 0; j < batch_size; j++) {
                int predicted_class = 0;
                float max_prob = predictions[j * layer_sizes.back()];

                for (int k = 1; k < layer_sizes.back(); k++) {
                    if (predictions[j * layer_sizes.back() + k] > max_prob) {
                        max_prob = predictions[j * layer_sizes.back() + k];
                        predicted_class = k;
                    }
                }

                if (predicted_class == test_labels[i + j]) {
                    correct++;
                }
            }
        }

        return 100.0 * correct / test_data.size();
    }

    vector<float> predict(const vector<float>& batch_inputs, size_t batch_size) {
        cl_int err;

        // Upload inputs
        err = queue.enqueueWriteBuffer(activation_buffers[0], CL_TRUE, 0,
                                     batch_inputs.size() * sizeof(float),
                                     batch_inputs.data());
        cl_check(err);

        // Forward pass (simplified version)
        for (size_t layer = 0; layer < layer_sizes.size() - 1; layer++) {
            // Set kernel arguments
            forward_kernel.setArg(0, activation_buffers[layer]);
            forward_kernel.setArg(1, weight_buffers[layer]);
            forward_kernel.setArg(2, bias_buffers[layer]);
            forward_kernel.setArg(3, activation_buffers[layer + 1]);
            forward_kernel.setArg(4, (int)batch_size);
            forward_kernel.setArg(5, layer_sizes[layer]);
            forward_kernel.setArg(6, layer_sizes[layer + 1]);

            // Execute kernel
            size_t global_work_size = batch_size * layer_sizes[layer + 1];
            err = queue.enqueueNDRangeKernel(forward_kernel, cl::NullRange,
                                           cl::NDRange(global_work_size), cl::NullRange);
            cl_check(err);

            // Apply activation
            if (layer < layer_sizes.size() - 2) {
                size_t activation_size = batch_size * layer_sizes[layer + 1];
                relu_kernel.setArg(0, activation_buffers[layer + 1]);
                relu_kernel.setArg(1, (int)activation_size);

                err = queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange,
                                               cl::NDRange(activation_size), cl::NullRange);
                cl_check(err);
            } else {
                // Apply softmax for output layer
                size_t max_work_group_size = static_cast<size_t>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
                size_t local_work_size = min(static_cast<size_t>(64), max_work_group_size);

                softmax_kernel.setArg(0, activation_buffers[layer + 1]);
                softmax_kernel.setArg(1, cl::Local(local_work_size * sizeof(float)));
                softmax_kernel.setArg(2, cl::Local(local_work_size * sizeof(float)));
                softmax_kernel.setArg(3, (int)batch_size);
                softmax_kernel.setArg(4, layer_sizes[layer + 1]);

                err = queue.enqueueNDRangeKernel(softmax_kernel, cl::NullRange,
                                               cl::NDRange(batch_size * local_work_size),
                                               cl::NDRange(local_work_size));
                cl_check(err);
            }
        }

        queue.finish();

        // Download results
        vector<float> results(batch_size * layer_sizes.back());
        err = queue.enqueueReadBuffer(activation_buffers.back(), CL_TRUE, 0,
                                    results.size() * sizeof(float), results.data());
        cl_check(err);

        return results;
    }
};

// Data loading functions with multiple path attempts
vector<vector<float>> load_images(const string& filename, int num_images) {
    // Try multiple possible paths
    vector<string> possible_paths = {
        filename,                           // Current directory
        "FMNIST/" + filename,              // FMNIST subdirectory
        "../FMNIST/" + filename,           // Parent/FMNIST subdirectory
        "../../FMNIST/" + filename         // Two levels up
    };

    ifstream file;
    string actual_path;

    // Try each path until one works
    for (const auto& path : possible_paths) {
        file.open(path, ios::binary);
        if (file.is_open()) {
            actual_path = path;
            cout << "Found images file at: " << path << endl;
            break;
        }
    }

    if (!file.is_open()) {
        cerr << "Error: Could not find file " << filename << " in any of these locations:" << endl;
        for (const auto& path : possible_paths) {
            cerr << "  - " << path << endl;
        }
        return {};
    }

    file.seekg(16);
    vector<vector<float>> images(num_images, vector<float>(784));
    vector<unsigned char> raw_data(num_images * 784);
    file.read(reinterpret_cast<char*>(raw_data.data()), num_images * 784);

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < 784; ++j) {
            images[i][j] = raw_data[i * 784 + j] / 255.0f;
        }
    }
    return images;
}

vector<int> load_labels(const string& filename, int num_labels) {
    // Try multiple possible paths
    vector<string> possible_paths = {
        filename,                           // Current directory
        "FMNIST/" + filename,              // FMNIST subdirectory
        "../FMNIST/" + filename,           // Parent/FMNIST subdirectory
        "../../FMNIST/" + filename         // Two levels up
    };

    ifstream file;
    string actual_path;

    // Try each path until one works
    for (const auto& path : possible_paths) {
        file.open(path, ios::binary);
        if (file.is_open()) {
            actual_path = path;
            cout << "Found labels file at: " << path << endl;
            break;
        }
    }

    if (!file.is_open()) {
        cerr << "Error: Could not find file " << filename << " in any of these locations:" << endl;
        for (const auto& path : possible_paths) {
            cerr << "  - " << path << endl;
        }
        return {};
    }

    file.seekg(8);
    vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = label;
    }
    return labels;
}

int main() {
    cout << "Fashion-MNIST MLP Classifier (OpenCL Optimized)" << endl;
    cout << "===============================================" << endl;

    vector<string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    cout << "\nClass Labels:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << i << ": " << class_names[i] << endl;
    }

    try {
        cout << "Loading training data..." << endl;
        auto train_images = load_images("train-images-idx3-ubyte", NOF_TRAIN);
        auto train_labels = load_labels("train-labels-idx1-ubyte", NOF_TRAIN);

        cout << "Loading test data..." << endl;
        auto test_images = load_images("t10k-images-idx3-ubyte", NOF_TEST);
        auto test_labels = load_labels("t10k-labels-idx1-ubyte", NOF_TEST);

        if (train_images.empty() || test_images.empty()) {
            cout << "Failed to load data. Please ensure Fashion-MNIST files are in the current directory." << endl;
            return 1;
        }

        cout << "Data loaded successfully!" << endl;
        cout << "Training samples: " << train_images.size() << endl;
        cout << "Test samples: " << test_images.size() << endl;

        vector<int> architecture = {784, 128, 64, 10};
        OpenCL_MLP network(architecture, LR);

        cout << "\nNetwork Architecture:" << endl;
        cout << "Input Layer: 784 neurons (28x28 pixels)" << endl;
        for (int i = 1; i < architecture.size() - 1; ++i) {
            cout << "Hidden Layer " << i << ": " << architecture[i] << " neurons (ReLU)" << endl;
        }
        cout << "Output Layer: 10 neurons (Softmax)" << endl;
        cout << "Batch size: " << BATCH_SIZE << " (optimized for GPU)" << endl;

        cout << "\nStarting training..." << endl;

        auto start = high_resolution_clock::now();
        network.train(train_images, train_labels, test_images, test_labels, EPOCHS);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(stop - start);
        double seconds = duration.count() / 1000.0;

        cout << "\nTraining completed!" << endl;
        cout << "Total training time: " << fixed << setprecision(3)
             << seconds << " seconds" << endl;
        cout << "Average time per epoch: " << fixed << setprecision(3)
             << seconds / EPOCHS << " seconds" << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
/*
Fashion-MNIST MLP Classifier (OpenCL Optimized)
===============================================

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
Found images file at: train-images-idx3-ubyte
Found labels file at: train-labels-idx1-ubyte
Loading test data...
Found images file at: t10k-images-idx3-ubyte
Found labels file at: t10k-labels-idx1-ubyte
Data loaded successfully!
Training samples: 5000
Test samples: 500
Using device: NVIDIA GeForce GTX 1650
Max compute units: 16
Max work group size: 1024
OpenCL kernels compiled successfully!

Network Architecture:
Input Layer: 784 neurons (28x28 pixels)
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 64 neurons (ReLU)
Output Layer: 10 neurons (Softmax)
Batch size: 64 (optimized for GPU)

Starting training...
Starting OpenCL training with optimized kernels...
Epoch 1/50 - Loss: 1.9893 - Accuracy: 62.80% - Time: 79ms
Epoch 6/50 - Loss: 0.7457 - Accuracy: 73.60% - Time: 70ms
Epoch 11/50 - Loss: 0.6030 - Accuracy: 75.00% - Time: 66ms
Epoch 16/50 - Loss: 0.5360 - Accuracy: 73.60% - Time: 66ms
Epoch 21/50 - Loss: 0.4884 - Accuracy: 81.00% - Time: 63ms
Epoch 26/50 - Loss: 0.4669 - Accuracy: 75.40% - Time: 66ms
Epoch 31/50 - Loss: 0.4406 - Accuracy: 77.20% - Time: 65ms
Epoch 36/50 - Loss: 0.4226 - Accuracy: 81.00% - Time: 64ms
Epoch 41/50 - Loss: 0.4104 - Accuracy: 73.40% - Time: 65ms
Epoch 46/50 - Loss: 0.4083 - Accuracy: 75.60% - Time: 63ms

Training completed!
Total training time: 3.512 seconds
Average time per epoch: 0.070 seconds
*/