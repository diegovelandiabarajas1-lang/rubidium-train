// ============================================================
// RUBIDIUM - CPU Parallel Matrix Operations (OpenMP)
// ============================================================
#pragma once
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <random>
#include <omp.h>

// ============================================================
// MATRIX CLASS (Row-major)
// ============================================================
struct Mat {
    int rows, cols;
    std::vector<float> data;

    Mat() : rows(0), cols(0) {}
    Mat(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    Mat(int r, int c, float val) : rows(r), cols(c), data(r * c, val) {}

    float& operator()(int i, int j) { return data[i * cols + j]; }
    float operator()(int i, int j) const { return data[i * cols + j]; }
    float* row(int i) { return data.data() + i * cols; }
    const float* row(int i) const { return data.data() + i * cols; }

    int size() const { return rows * cols; }
    void zero() { std::fill(data.begin(), data.end(), 0.0f); }
    void fill(float v) { std::fill(data.begin(), data.end(), v); }

    void randn(float std = 1.0f) {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, std);
        for (auto &v : data) v = dist(gen);
    }
};

// ============================================================
// PARALLEL MATRIX OPERATIONS (OpenMP)
// ============================================================
namespace cpuops {

// C = A * B (GEMM) - Parallel by row
inline void matmul(Mat &C, const Mat &A, const Mat &B, float alpha = 1.0f, float beta = 0.0f) {
    int M = A.rows, K = A.cols, N = B.cols;
    C.rows = M; C.cols = N;
    if ((int)C.data.size() != M * N) C.data.resize(M * N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A(i, k) * B(k, j);
            C(i, j) = alpha * sum + beta * C(i, j);
        }
    }
}

// C += A * B^T (GEMM transpose B)
inline void matmul_tB(Mat &C, const Mat &A, const Mat &B, float alpha = 1.0f) {
    int M = A.rows, K = A.cols, N = B.rows;
    C.rows = M; C.cols = N;
    if ((int)C.data.size() != M * N) C.data.resize(M * N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A(i, k) * B(j, k);
            C(i, j) += alpha * sum;
        }
    }
}

// C += A^T * B
inline void matmul_tA(Mat &C, const Mat &A, const Mat &B, float alpha = 1.0f) {
    int M = A.cols, K = A.rows, N = B.cols;
    C.rows = M; C.cols = N;
    if ((int)C.data.size() != M * N) C.data.resize(M * N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A(k, i) * B(k, j);
            C(i, j) += alpha * sum;
        }
    }
}

// LayerNorm forward
inline void layer_norm(Mat &out, Mat &mean, Mat &inv_std,
                       const Mat &x, const Mat &w, const Mat &b, float eps = 1e-5f) {
    int N = x.rows, D = x.cols;
    out = Mat(N, D);
    mean = Mat(N, 1);
    inv_std = Mat(N, 1);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float m = 0.0f;
        for (int j = 0; j < D; j++) m += x(i, j);
        m /= D;
        mean(i, 0) = m;

        float v = 0.0f;
        for (int j = 0; j < D; j++) {
            float d = x(i, j) - m;
            v += d * d;
        }
        float inv = 1.0f / std::sqrt(v / D + eps);
        inv_std(i, 0) = inv;

        for (int j = 0; j < D; j++)
            out(i, j) = w(0, j) * (x(i, j) - m) * inv + b(0, j);
    }
}

// LayerNorm backward
inline void layer_norm_backward(Mat &dx, Mat &dw, Mat &db,
                                const Mat &dout, const Mat &x, const Mat &w,
                                const Mat &mean, const Mat &inv_std) {
    int N = x.rows, D = x.cols;
    dx = Mat(N, D);
    dw = Mat(1, D);
    db = Mat(1, D);

    #pragma omp parallel for reduction(+:dw.data[:D], db.data[:D]) schedule(static)
    for (int i = 0; i < N; i++) {
        float m = mean(i, 0);
        float inv = inv_std(i, 0);
        float ts1 = 0.0f, ts2 = 0.0f;
        for (int j = 0; j < D; j++) {
            float dw_j = dout(i, j) * w(0, j);
            ts1 += dw_j;
            ts2 += dw_j * (x(i, j) - m);
        }
        for (int j = 0; j < D; j++) {
            float dw_j = dout(i, j) * w(0, j);
            dx(i, j) = inv * (dw_j - (ts1 + (x(i, j) - m) * inv * ts2) / D);
            #pragma omp atomic
            dw(0, j) += dout(i, j) * (x(i, j) - m) * inv;
            #pragma omp atomic
            db(0, j) += dout(i, j);
        }
    }
}

// ReLU forward
inline void relu(Mat &out, const Mat &x) {
    out = Mat(x.rows, x.cols);
    #pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
        out.data[i] = std::max(0.0f, x.data[i]);
}

// ReLU backward
inline void relu_backward(Mat &dx, const Mat &dout, const Mat &x) {
    dx = Mat(x.rows, x.cols);
    #pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
        dx.data[i] = x.data[i] > 0.0f ? dout.data[i] : 0.0f;
}

// Softmax forward (per row)
inline void softmax(Mat &out, const Mat &x) {
    int N = x.rows, C = x.cols;
    out = Mat(N, C);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float mx = -1e30f;
        for (int j = 0; j < C; j++) mx = std::max(mx, x(i, j));
        float s = 0.0f;
        for (int j = 0; j < C; j++) {
            out(i, j) = std::exp(x(i, j) - mx);
            s += out(i, j);
        }
        for (int j = 0; j < C; j++) out(i, j) /= s;
    }
}

// Softmax backward
inline void softmax_backward(Mat &dx, const Mat &dout, const Mat &out) {
    int N = out.rows, C = out.cols;
    dx = Mat(N, C);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float dot = 0.0f;
        for (int j = 0; j < C; j++) dot += dout(i, j) * out(i, j);
        for (int j = 0; j < C; j++) dx(i, j) = out(i, j) * (dout(i, j) - dot);
    }
}

// Cross entropy forward
inline float cross_entropy(const Mat &logits, const std::vector<int> &targets) {
    int N = logits.rows, V = logits.cols;
    float loss = 0.0f;

    #pragma omp parallel for reduction(+:loss) schedule(static)
    for (int i = 0; i < N; i++) {
        float mx = -1e30f;
        for (int j = 0; j < V; j++) mx = std::max(mx, logits(i, j));
        float s = 0.0f;
        for (int j = 0; j < V; j++) s += std::exp(logits(i, j) - mx);
        float log_prob = logits(i, targets[i]) - mx - std::log(s + 1e-10f);
        loss -= log_prob;
    }
    return loss / N;
}

// Cross entropy backward
inline void cross_entropy_backward(Mat &d_logits, const Mat &logits,
                                    const std::vector<int> &targets) {
    int N = logits.rows, V = logits.cols;
    d_logits = Mat(N, V);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float mx = -1e30f;
        for (int j = 0; j < V; j++) mx = std::max(mx, logits(i, j));
        float s = 0.0f;
        for (int j = 0; j < V; j++) s += std::exp(logits(i, j) - mx);
        for (int j = 0; j < V; j++) {
            float prob = std::exp(logits(i, j) - mx) / s;
            d_logits(i, j) = (prob - (j == targets[i] ? 1.0f : 0.0f)) / N;
        }
    }
}

// Dropout forward
inline void dropout(Mat &out, const Mat &x, float p, std::mt19937 &gen) {
    out = Mat(x.rows, x.cols);
    float scale = 1.0f / (1.0f - p);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    #pragma omp parallel for
    for (int i = 0; i < x.size(); i++) {
        float r = dist(gen);
        out.data[i] = r < p ? 0.0f : x.data[i] * scale;
    }
}

// Embedding forward
inline void embedding(Mat &out, const Mat &weight, const std::vector<int> &indices) {
    int T = indices.size(), D = weight.cols;
    out = Mat(T, D);

    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        int id = indices[t];
        for (int d = 0; d < D; d++)
            out(t, d) = weight(id, d);
    }
}

// Embedding backward
inline void embedding_backward(Mat &d_weight, const Mat &d_out,
                                const std::vector<int> &indices, int V) {
    int T = indices.size(), D = d_out.cols;
    d_weight = Mat(V, D);

    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        int id = indices[t];
        for (int d = 0; d < D; d++)
            #pragma omp atomic
            d_weight(id, d) += d_out(t, d);
    }
}

// AdamW step
inline void adamw_step(Mat &p, const Mat &g, Mat &m, Mat &v,
                       float lr, float b1, float b2, float eps, float wd, int t) {
    int n = p.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float gi = g.data[i] + wd * p.data[i];
        m.data[i] = b1 * m.data[i] + (1.0f - b1) * gi;
        v.data[i] = b2 * v.data[i] + (1.0f - b2) * gi * gi;
        float mh = m.data[i] / (1.0f - std::pow(b1, (float)t));
        float vh = v.data[i] / (1.0f - std::pow(b2, (float)t));
        p.data[i] -= lr * mh / (std::sqrt(vh) + eps);
    }
}

// Gradient clipping
inline float clip_gradients(Mat &g, float max_norm) {
    float norm = 0.0f;
    #pragma omp parallel for reduction(+:norm)
    for (int i = 0; i < g.size(); i++) norm += g.data[i] * g.data[i];
    norm = std::sqrt(norm);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        #pragma omp parallel for
        for (int i = 0; i < g.size(); i++) g.data[i] *= scale;
    }
    return norm;
}

} // namespace cpuops
