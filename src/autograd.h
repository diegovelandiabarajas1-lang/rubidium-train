// ============================================================
// RUBIDIUM TRANSFORMER - AUTOGRADE ENGINE
// Tensor struct with GPU memory, backward pass, gradient accumulation
// Arena allocator (no garbage collection)
// ============================================================
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <functional>
#include <cstdlib>
#include <cstdio>

// ============================================================
// ARENA ALLOCATOR
// ============================================================
struct Arena {
    char *base;
    size_t capacity;
    size_t used;

    Arena(size_t cap = 512 * 1024 * 1024) : capacity(cap), used(0) {
        base = (char *)malloc(capacity);
        if (!base) { fprintf(stderr, "Arena alloc failed: %zu MB\n", capacity / 1024 / 1024); exit(1); }
    }
    ~Arena() { free(base); }

    void *alloc(size_t size, size_t alignment = 256) {
        size_t pad = (alignment - (used % alignment)) % alignment;
        if (used + pad + size > capacity) {
            fprintf(stderr, "Arena overflow: %zu + %zu > %zu\n", used + pad, size, capacity);
            exit(1);
        }
        void *ptr = base + used + pad;
        used += pad + size;
        return ptr;
    }

    void reset() { used = 0; }
};

// ============================================================
// TENSOR
// ============================================================
struct Tensor {
    float *data;       // GPU pointer
    float *grad;       // GPU pointer (nullptr if no grad)
    int ndim;
    int shape[4];      // max 4D
    int stride[4];
    int size;          // total elements
    bool requires_grad;
    bool owns_data;    // true if we allocated data

    Tensor() : data(nullptr), grad(nullptr), ndim(0), size(0),
               requires_grad(false), owns_data(false) {
        memset(shape, 0, sizeof(shape));
        memset(stride, 0, sizeof(stride));
    }

    // Create from existing GPU pointer (does NOT own memory)
    static Tensor from_gpu(float *ptr, int ndim, const int *shape, bool grad = false) {
        Tensor t;
        t.data = ptr;
        t.ndim = ndim;
        memcpy(t.shape, shape, ndim * sizeof(int));
        t.size = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            t.stride[i] = (i < ndim - 1) ? t.stride[i + 1] * t.shape[i + 1] : 1;
            t.size *= shape[i];
        }
        if (grad) {
            t.requires_grad = true;
            cudaMalloc(&t.grad, t.size * sizeof(float));
            cudaMemset(t.grad, 0, t.size * sizeof(float));
        }
        t.owns_data = false;
        return t;
    }

    // Allocate new GPU memory
    static Tensor create(int ndim, const int *shape, bool grad = false) {
        Tensor t;
        t.ndim = ndim;
        memcpy(t.shape, shape, ndim * sizeof(int));
        t.size = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            t.stride[i] = (i < ndim - 1) ? t.stride[i + 1] * t.shape[i + 1] : 1;
            t.size *= shape[i];
        }
        cudaMalloc(&t.data, t.size * sizeof(float));
        cudaMemset(t.data, 0, t.size * sizeof(float));
        if (grad) {
            t.requires_grad = true;
            cudaMalloc(&t.grad, t.size * sizeof(float));
            cudaMemset(t.grad, 0, t.size * sizeof(float));
        }
        t.owns_data = true;
        return t;
    }

    // Create scalar
    static Tensor scalar(float value, bool grad = false) {
        int shape[] = {1};
        Tensor t = create(1, shape, grad);
        cudaMemcpy(t.data, &value, sizeof(float), cudaMemcpyHostToDevice);
        return t;
    }

    void zero_grad() {
        if (grad) cudaMemset(grad, 0, size * sizeof(float));
    }

    void free() {
        if (owns_data && data) { cudaFree(data); data = nullptr; }
        if (grad) { cudaFree(grad); grad = nullptr; }
    }
};

// ============================================================
// COMPUTATION GRAPH NODE
// ============================================================
enum OpType {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MATMUL,
    OP_RELU, OP_SOFTMAX, OP_LAYER_NORM, OP_CROSS_ENTROPY,
    OP_EMBEDDING, OP_LINEAR, OP_DROPOUT,
    OP_SUM, OP_MEAN, OP_RESHAPE, OP_TRANSPOSE
};

struct Node {
    OpType op;
    Tensor output;
    std::vector<Tensor> inputs;   // input tensors
    std::vector<void *> extra;    // extra params (weights, etc.)

    std::function<void(const float *)> backward_fn;

    Node(OpType op, const Tensor &out) : op(op), output(out) {}
};

// ============================================================
// AUTOGRADE ENGINE
// ============================================================
struct AutogradEngine {
    std::vector<Node *> graph;
    Arena arena;

    AutogradEngine(size_t arena_size = 512 * 1024 * 1024) : arena(arena_size) {}

    ~AutogradEngine() {
        for (auto *n : graph) {
            n->output.free();
            for (auto &inp : n->inputs) inp.free();
            delete n;
        }
    }

    Node *add_node(OpType op, const Tensor &output) {
        Node *n = new Node(op, output);
        graph.push_back(n);
        return n;
    }

    void backward(Tensor &loss) {
        // Simple reverse-mode autodiff
        // We assume loss is a scalar
        // Set initial gradient to 1.0
        float one = 1.0f;
        cudaMemcpy(loss.grad, &one, sizeof(float), cudaMemcpyHostToDevice);

        // Process nodes in reverse order
        for (int i = (int)graph.size() - 1; i >= 0; i--) {
            Node *n = graph[i];
            if (n->backward_fn) {
                n->backward_fn(n->output.grad);
            }
        }
    }

    void zero_grads() {
        for (auto *n : graph) {
            if (n->output.requires_grad) n->output.zero_grad();
        }
    }

    void clear() {
        for (auto *n : graph) {
            n->output.free();
            for (auto &inp : n->inputs) inp.free();
            delete n;
        }
        graph.clear();
        arena.reset();
    }
};
