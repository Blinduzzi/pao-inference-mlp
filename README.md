# PAO MLP Inference Optimization

**Multilayer Perceptron Training Acceleration on Fashion-MNIST Dataset**

This project implements a progressive optimization approach for MLP inference in C++, starting from a single-threaded naive implementation and applying various optimization techniques to achieve significant performance improvements.

## Project Overview

The goal is to explore different optimization strategies for neural network training and inference, demonstrating the computational benefits of parallelization, vectorization, and GPU acceleration. The project uses Fashion-MNIST dataset for classification tasks with a four-layer neural network architecture.

### Network Architecture
- **Input Layer**: 784 neurons (28×28 pixel Fashion-MNIST images)
- **Hidden Layer 1**: 128 neurons (ReLU activation)  
- **Hidden Layer 2**: 64 neurons (ReLU activation)
- **Output Layer**: 10 neurons (Softmax activation)

### Dataset
- **Training**: 5,000 Fashion-MNIST samples
- **Testing**: 500 Fashion-MNIST samples
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Project Structure

```
pao-inference-mlp/
├── naive_main.cpp          # Single-threaded baseline implementation
├── openmp_main.cpp         # OpenMP parallelized version  
├── avx_main.cpp           # AVX2 vectorized + OpenMP implementation
├── opencl_main.cpp        # OpenCL GPU-accelerated version
├── FMNIST/               # Fashion-MNIST dataset directory
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── docs/
│   ├── sota/              # State-of-the-Art paper selection
│   ├── PAO_MLP_F-MNIST_Blindu_AndreiSamuel_411-ACES.pdf # Documentation
└── README.md
```

## Optimization Strategies

### 1. **Naive Implementation** (`naive_main.cpp`)
- Single-threaded baseline
- No compiler optimizations (-O0)
- Sequential forward/backward propagation
- Basic matrix operations without vectorization

**Features:**
- Xavier weight initialization
- Standard backpropagation algorithm
- Individual sample processing

### 2. **Compiler Optimization** (O3 Flag)
- Same code with `-O3` compiler optimization
- Minimal performance improvement (1.009× speedup)

### 3. **OpenMP Parallelization** (`openmp_main.cpp`)
- Multi-threaded training with 8 OpenMP threads
- Mini-batch processing approach
- Thread-safe gradient computation
- Master-slave weight update pattern

**Features:**
- Parallel forward/backward propagation
- Independent thread-local variables
- Averaged gradient updates
- Eliminated race conditions

### 4. **AVX2 Vectorization** (`avx_main.cpp`)
- Combined OpenMP + AVX2 SIMD optimization
- 32-byte aligned memory structures (`AlignedMatrix`, `AlignedVector`)
- Vectorized mathematical operations
- 4-wide double-precision SIMD processing

**Key Optimizations:**
- `_mm256_fmadd_pd` for fused multiply-add operations
- `_mm256_load_pd/store_pd` for vectorized memory access
- `_mm256_max_pd` for parallel ReLU computations
- Cache-optimized memory layouts

### 5. **OpenCL GPU Acceleration** (`opencl_main.cpp`)
- GPU-based parallel computation
- Optimized kernel design for neural network operations
- Efficient memory management patterns
- Batch processing for improved GPU utilization

**Features:**
- Fast relaxed math operations (`-cl-fast-relaxed-math`)
- Memory coalescing patterns
- Local memory optimization for reductions
- Persistent GPU-resident data structures

## Performance Results

| Implementation | Training Time (s) | Time/Epoch (s) | Accuracy | Speedup |
|----------------|-------------------|----------------|----------|---------|
| **Naive**      | 937.104          | 18.742         | 87.0%    | 1×      |
| **O3**         | 928.242          | 18.565         | 86.5%    | 1.009×  |
| **OpenMP**     | 86.529           | 1.731          | 85.6%    | 10.83×  |
| **AVX2**       | 30.322           | 0.606          | 85.8%    | 30.91×  |
| **OpenCL**     | 4.207            | 0.084          | 86.4%    | **222.75×** |

*Test Environment: Intel Core i5-9300H CPU @ 2.40GHz, NVIDIA GeForce GTX 1650*

## Key Findings

### Performance Hierarchy
1. **Compiler optimizations** provide negligible improvements for this workload
2. **Multi-threading** delivers significant speedup (10.8×) through CPU parallelization
3. **SIMD vectorization** adds substantial acceleration (30.9×) via AVX2 instructions
4. **GPU acceleration** achieves significant performance gains (222×) through massive parallelism

### Accuracy Consistency
- Minimal accuracy degradation across optimizations (87% → 85.6%)
- Optimization techniques don't compromise learning capacity
- All implementations maintain competitive classification performance

## Build Instructions

### Prerequisites
- C++ compiler with C++11 support
- OpenMP support
- AVX2-capable processor (for AVX2 version)
- OpenCL SDK and compatible GPU (for OpenCL version)

### Compilation Commands

```bash
# Naive implementation
g++ -O0 naive_main.cpp -o naive_main

# Compiler optimized
g++ -O3 naive_main.cpp -o optimized_main

# OpenMP version
g++ -fopenmp -O3 openmp_main.cpp -o openmp_main

# AVX2 + OpenMP version
g++ -fopenmp -mavx2 -mfma -O3 -march=native avx_main.cpp -o avx_main

# OpenCL version
g++ -O3 opencl_main.cpp -lOpenCL -o opencl_main
```

## Usage

1. **Download Fashion-MNIST dataset** and place in `FMNIST/` directory
2. **Select implementation** based on available hardware
3. **Run the executable**:

```bash
# Example: Run AVX2 optimized version
./avx_main
```

**Expected Output:**
```
Fashion-MNIST MLP Classifier (Fixed AVX2 + Aligned Memory)
===========================================================

Network Architecture:
Input Layer: 784 neurons (28x28 pixels)
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 64 neurons (ReLU)
Output Layer: 10 neurons (Softmax)
Using 8 OpenMP threads with AVX2 optimization

Training completed!
Total training time: 30.322 seconds
Average time per epoch: 0.606 seconds
Final Test Accuracy: 85.80%
```

## Technical Contributions

### Memory Optimization
- Aligned memory allocation for SIMD compatibility
- Elimination of nested vector structures
- Cache-friendly data layouts
- Reduced memory fragmentation

### Parallel Computing
- Thread-safe gradient computation
- Load-balanced work distribution
- Efficient synchronization patterns
- GPU memory management optimization

### Algorithm Implementation
- Robust backpropagation implementation
- Numerical stability improvements
- Batch processing strategies
- Activation function optimizations

## References

- Fashion-MNIST Dataset: [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- Intel AVX2 Intrinsics: [Intel Developer Zone](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- OpenCL Specification: [Khronos Group](https://www.khronos.org/opencl/)
- SOTA in project documentation

## Author

**Eng. Blîndu Andrei-Samuel**  
National University of Science and Technology POLITEHNICA Bucharest  
Faculty of Electronics, Telecommunications and Information Technology

---

*This project demonstrates the evolution from basic single-threaded computation through multiple build iterations with progressive optimizations, showcasing the critical importance of hardware-software co-design in modern machine learning applications.*
