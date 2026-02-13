#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            throw std::runtime_error(std::string("CUDA error: ") +                  \
                                     cudaGetErrorString(err__) +                     \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                            \
    } while (0)

struct HyperParams {
    int n_embd = 16;
    int n_head = 4;
    int n_layer = 1;
    int block_size = 8;
    int head_dim = 4;
};

struct TrainConfig {
    int num_steps = 500;
    int val_every = 100;
    int val_docs = 20;
    int num_samples = 20;
    int top_k = 5;
    int seed = 42;
    float temperature = 0.6f;

    float learning_rate = 1e-2f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps_adam = 1e-8f;
    float weight_decay = 1e-4f;
    float max_grad_norm = 1.0f;
};

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> w;
    std::vector<float> g;
    std::vector<float> m;
    std::vector<float> v;

    Matrix() = default;

    Matrix(int rows_, int cols_, float stddev, std::mt19937& rng)
        : rows(rows_),
          cols(cols_),
          w(static_cast<size_t>(rows_) * static_cast<size_t>(cols_)),
          g(w.size(), 0.0f),
          m(w.size(), 0.0f),
          v(w.size(), 0.0f) {
        if (stddev == 0.0f) {
            std::fill(w.begin(), w.end(), 0.0f);
            return;
        }
        std::normal_distribution<float> dist(0.0f, stddev);
        for (float& x : w) {
            x = dist(rng);
        }
    }
};

struct LayerParams {
    Matrix attn_wq;
    Matrix attn_wk;
    Matrix attn_wv;
    Matrix attn_wo;
    Matrix mlp_fc1;
    Matrix mlp_fc2;

    LayerParams() = default;

    LayerParams(const HyperParams& hp, std::mt19937& rng)
        : attn_wq(hp.n_embd, hp.n_embd, 0.02f, rng),
          attn_wk(hp.n_embd, hp.n_embd, 0.02f, rng),
          attn_wv(hp.n_embd, hp.n_embd, 0.02f, rng),
          attn_wo(hp.n_embd, hp.n_embd, 0.0f, rng),
          mlp_fc1(4 * hp.n_embd, hp.n_embd, 0.02f, rng),
          mlp_fc2(hp.n_embd, 4 * hp.n_embd, 0.0f, rng) {}
};

struct Model {
    HyperParams hp;
    int vocab_size = 0;
    Matrix wte;
    Matrix wpe;
    std::vector<LayerParams> layers;

    Model(int vocab_size_, const HyperParams& hp_, std::mt19937& rng)
        : hp(hp_),
          vocab_size(vocab_size_),
          wte(vocab_size_, hp_.n_embd, 0.02f, rng),
          wpe(hp_.block_size, hp_.n_embd, 0.02f, rng) {
        layers.reserve(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            layers.emplace_back(hp, rng);
        }
    }

    template <typename Fn>
    void for_each_matrix(Fn&& fn) {
        fn(wte);
        fn(wpe);
        for (LayerParams& layer : layers) {
            fn(layer.attn_wq);
            fn(layer.attn_wk);
            fn(layer.attn_wv);
            fn(layer.attn_wo);
            fn(layer.mlp_fc1);
            fn(layer.mlp_fc2);
        }
    }

    template <typename Fn>
    void for_each_matrix(Fn&& fn) const {
        fn(wte);
        fn(wpe);
        for (const LayerParams& layer : layers) {
            fn(layer.attn_wq);
            fn(layer.attn_wk);
            fn(layer.attn_wv);
            fn(layer.attn_wo);
            fn(layer.mlp_fc1);
            fn(layer.mlp_fc2);
        }
    }

    void zero_grads() {
        for_each_matrix([](Matrix& mat) {
            std::fill(mat.g.begin(), mat.g.end(), 0.0f);
        });
    }

    size_t num_params() const {
        size_t n = 0;
        for_each_matrix([&](const Matrix& mat) { n += mat.w.size(); });
        return n;
    }
};

struct DataBundle {
    std::vector<std::string> docs;
    std::vector<std::string> train_docs;
    std::vector<std::string> val_docs;
    std::unordered_map<char, int> stoi;
    std::vector<std::string> itos;
    int bos = 0;
};

struct LayerStepCache {
    std::vector<float> x_resid_attn;
    std::vector<float> x_norm1;
    float inv_rms1 = 0.0f;

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<std::vector<float>> attn_weights;

    std::vector<float> x_attn;
    std::vector<float> x_after_attn;
    std::vector<float> x_norm2;
    float inv_rms2 = 0.0f;

    std::vector<float> fc1_pre;
    std::vector<float> fc1_act;
    std::vector<float> x_out;

    LayerStepCache() = default;

    explicit LayerStepCache(const HyperParams& hp)
        : x_resid_attn(hp.n_embd, 0.0f),
          x_norm1(hp.n_embd, 0.0f),
          q(hp.n_embd, 0.0f),
          k(hp.n_embd, 0.0f),
          v(hp.n_embd, 0.0f),
          attn_weights(static_cast<size_t>(hp.n_head)),
          x_attn(hp.n_embd, 0.0f),
          x_after_attn(hp.n_embd, 0.0f),
          x_norm2(hp.n_embd, 0.0f),
          fc1_pre(4 * hp.n_embd, 0.0f),
          fc1_act(4 * hp.n_embd, 0.0f),
          x_out(hp.n_embd, 0.0f) {}
};

struct StepCache {
    int token_id = -1;
    int target_id = -1;
    std::vector<float> x_tokpos;
    std::vector<float> x0;
    float inv_rms0 = 0.0f;
    std::vector<LayerStepCache> layers;
    std::vector<float> x_final;
    std::vector<float> logits;
    std::vector<float> probs;

    StepCache() = default;

    StepCache(const HyperParams& hp, int vocab_size)
        : x_tokpos(hp.n_embd, 0.0f),
          x0(hp.n_embd, 0.0f),
          layers(),
          x_final(hp.n_embd, 0.0f),
          logits(vocab_size, 0.0f),
          probs(vocab_size, 0.0f) {
        layers.reserve(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            layers.emplace_back(hp);
        }
    }
};

struct KVLayerCache {
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

__global__ void matvec_kernel(const float* w, const float* x, float* y, int out, int in) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out) {
        return;
    }
    float acc = 0.0f;
    int base = row * in;
    for (int i = 0; i < in; ++i) {
        acc += w[base + i] * x[i];
    }
    y[row] = acc;
}

__global__ void matvec_t_kernel(const float* w, const float* dy, float* dx, int out, int in) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= in) {
        return;
    }
    float acc = 0.0f;
    for (int row = 0; row < out; ++row) {
        acc += w[row * in + col] * dy[row];
    }
    dx[col] = acc;
}

__global__ void outer_add_kernel(float* dw, const float* dy, const float* x, int out, int in, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= out || col >= in) {
        return;
    }
    dw[row * in + col] += (dy[row] * x[col]) * scale;
}

__global__ void adamw_kernel(
    float* w,
    float* g,
    float* m,
    float* v,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float one_minus_b1_prod,
    float one_minus_b2_prod,
    float grad_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float grad = g[idx] * grad_scale;
    float mi = beta1 * m[idx] + (1.0f - beta1) * grad;
    float vi = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = mi;
    v[idx] = vi;
    float m_hat = mi / one_minus_b1_prod;
    float v_hat = vi / one_minus_b2_prod;
    w[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w[idx]);
    g[idx] = 0.0f;
}

class CudaOps {
public:
    CudaOps() = default;
    ~CudaOps() {
        free_if_needed(d_w_);
        free_if_needed(d_a_);
        free_if_needed(d_b_);
        free_if_needed(d_c_);
        free_if_needed(d_d_);
    }

    void matvec(const float* w, const float* x, float* y, int out, int in) {
        ensure(d_w_, cap_w_, out * in);
        ensure(d_a_, cap_a_, in);
        ensure(d_b_, cap_b_, out);

        CUDA_CHECK(cudaMemcpy(d_w_, w, sizeof(float) * out * in, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_, x, sizeof(float) * in, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (out + threads - 1) / threads;
        matvec_kernel<<<blocks, threads>>>(d_w_, d_a_, d_b_, out, in);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y, d_b_, sizeof(float) * out, cudaMemcpyDeviceToHost));
    }

    void matvec_t(const float* w, const float* dy, float* dx, int out, int in) {
        ensure(d_w_, cap_w_, out * in);
        ensure(d_a_, cap_a_, out);
        ensure(d_b_, cap_b_, in);

        CUDA_CHECK(cudaMemcpy(d_w_, w, sizeof(float) * out * in, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_, dy, sizeof(float) * out, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (in + threads - 1) / threads;
        matvec_t_kernel<<<blocks, threads>>>(d_w_, d_a_, d_b_, out, in);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(dx, d_b_, sizeof(float) * in, cudaMemcpyDeviceToHost));
    }

    void outer_add(float* dw, const float* dy, const float* x, int out, int in, float scale) {
        ensure(d_w_, cap_w_, out * in);
        ensure(d_a_, cap_a_, out);
        ensure(d_b_, cap_b_, in);

        CUDA_CHECK(cudaMemcpy(d_w_, dw, sizeof(float) * out * in, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_, dy, sizeof(float) * out, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_, x, sizeof(float) * in, cudaMemcpyHostToDevice));

        dim3 threads(16, 16);
        dim3 blocks((in + threads.x - 1) / threads.x, (out + threads.y - 1) / threads.y);
        outer_add_kernel<<<blocks, threads>>>(d_w_, d_a_, d_b_, out, in, scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(dw, d_w_, sizeof(float) * out * in, cudaMemcpyDeviceToHost));
    }

    void adamw_update(
        float* w,
        float* g,
        float* m,
        float* v,
        int n,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float one_minus_b1_prod,
        float one_minus_b2_prod,
        float grad_scale) {
        ensure(d_a_, cap_a_, n);
        ensure(d_b_, cap_b_, n);
        ensure(d_c_, cap_c_, n);
        ensure(d_d_, cap_d_, n);

        CUDA_CHECK(cudaMemcpy(d_a_, w, sizeof(float) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_, g, sizeof(float) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c_, m, sizeof(float) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_d_, v, sizeof(float) * n, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        adamw_kernel<<<blocks, threads>>>(
            d_a_, d_b_, d_c_, d_d_, n, lr, beta1, beta2, eps, weight_decay, one_minus_b1_prod,
            one_minus_b2_prod, grad_scale);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(w, d_a_, sizeof(float) * n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m, d_c_, sizeof(float) * n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v, d_d_, sizeof(float) * n, cudaMemcpyDeviceToHost));
        std::fill(g, g + n, 0.0f);
    }

private:
    static void free_if_needed(float*& ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    static void ensure(float*& ptr, int& cap, int n) {
        if (cap >= n) {
            return;
        }
        free_if_needed(ptr);
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(float) * n));
        cap = n;
    }

    float* d_w_ = nullptr;
    float* d_a_ = nullptr;
    float* d_b_ = nullptr;
    float* d_c_ = nullptr;
    float* d_d_ = nullptr;

    int cap_w_ = 0;
    int cap_a_ = 0;
    int cap_b_ = 0;
    int cap_c_ = 0;
    int cap_d_ = 0;
};

static std::string trim_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

static void ensure_input_file() {
    namespace fs = std::filesystem;
    if (fs::exists("input.txt")) {
        return;
    }
#ifdef _WIN32
    const char* cmd =
        "powershell -NoProfile -Command \""
        "$u='https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt';"
        "Invoke-WebRequest -Uri $u -OutFile 'input.txt' -UseBasicParsing\"";
    int rc = std::system(cmd);
    if (rc != 0 || !fs::exists("input.txt")) {
        throw std::runtime_error("failed to download input.txt");
    }
#else
    throw std::runtime_error("input.txt missing and auto-download only implemented for Windows");
#endif
}

static DataBundle load_data(std::mt19937& rng) {
    ensure_input_file();

    std::ifstream f("input.txt");
    if (!f) {
        throw std::runtime_error("failed to open input.txt");
    }

    DataBundle data;
    std::string line;
    while (std::getline(f, line)) {
        std::string s = trim_copy(line);
        if (!s.empty()) {
            data.docs.push_back(s);
        }
    }
    if (data.docs.empty()) {
        throw std::runtime_error("input.txt is empty");
    }

    std::shuffle(data.docs.begin(), data.docs.end(), rng);
    size_t split = static_cast<size_t>(0.9 * static_cast<double>(data.docs.size()));
    if (split == 0 && data.docs.size() > 1) {
        split = 1;
    }
    if (split >= data.docs.size() && data.docs.size() > 1) {
        split = data.docs.size() - 1;
    }
    data.train_docs.assign(data.docs.begin(), data.docs.begin() + static_cast<std::ptrdiff_t>(split));
    data.val_docs.assign(data.docs.begin() + static_cast<std::ptrdiff_t>(split), data.docs.end());

    std::array<bool, 256> seen{};
    for (const std::string& doc : data.docs) {
        for (unsigned char c : doc) {
            seen[c] = true;
        }
    }

    std::vector<char> chars;
    for (int i = 0; i < 256; ++i) {
        if (seen[static_cast<size_t>(i)]) {
            chars.push_back(static_cast<char>(i));
        }
    }
    std::sort(chars.begin(), chars.end());

    data.itos.clear();
    data.itos.push_back("<BOS>");
    data.bos = 0;
    for (char c : chars) {
        data.stoi[c] = static_cast<int>(data.itos.size());
        data.itos.push_back(std::string(1, c));
    }
    return data;
}

static std::vector<int> encode_doc(
    const std::string& doc,
    const std::unordered_map<char, int>& stoi,
    int bos_token) {
    std::vector<int> tokens;
    tokens.reserve(doc.size() + 2);
    tokens.push_back(bos_token);
    for (char c : doc) {
        auto it = stoi.find(c);
        if (it == stoi.end()) {
            throw std::runtime_error("document contains out-of-vocab character");
        }
        tokens.push_back(it->second);
    }
    tokens.push_back(bos_token);
    return tokens;
}

static void add_inplace(std::vector<float>& dst, const std::vector<float>& src) {
    if (dst.size() != src.size()) {
        throw std::runtime_error("add_inplace size mismatch");
    }
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

static void rmsnorm_forward(const std::vector<float>& x, std::vector<float>& y, float& inv_rms) {
    double ms = 0.0;
    for (float xi : x) {
        ms += static_cast<double>(xi) * static_cast<double>(xi);
    }
    ms /= static_cast<double>(x.size());
    inv_rms = 1.0f / std::sqrt(static_cast<float>(ms) + 1e-5f);
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] * inv_rms;
    }
}

static void rmsnorm_backward(
    const std::vector<float>& x,
    float inv_rms,
    const std::vector<float>& dy,
    std::vector<float>& dx) {
    if (x.size() != dy.size()) {
        throw std::runtime_error("rmsnorm_backward size mismatch");
    }
    double dot = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        dot += static_cast<double>(dy[i]) * static_cast<double>(x[i]);
    }
    float coeff = (inv_rms * inv_rms * inv_rms) / static_cast<float>(x.size());
    dx.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        dx[i] = dy[i] * inv_rms - x[i] * static_cast<float>(dot) * coeff;
    }
}

static void stable_softmax(const std::vector<float>& logits, std::vector<float>& probs) {
    float mx = *std::max_element(logits.begin(), logits.end());
    probs.resize(logits.size());
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - mx);
        sum_exp += probs[i];
    }
    float inv = 1.0f / static_cast<float>(sum_exp);
    for (float& p : probs) {
        p *= inv;
    }
}

static float cross_entropy_with_probs(const std::vector<float>& logits, int target_id, std::vector<float>& probs) {
    float mx = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (float l : logits) {
        sum_exp += std::exp(l - mx);
    }
    stable_softmax(logits, probs);
    return -(logits[target_id] - mx - std::log(sum_exp));
}

static void cross_entropy_grad(
    const std::vector<float>& probs,
    int target_id,
    float scale,
    std::vector<float>& dlogits) {
    dlogits = probs;
    for (float& v : dlogits) {
        v *= scale;
    }
    dlogits[target_id] -= scale;
}

static void linear_forward(
    const Matrix& mat,
    const std::vector<float>& x,
    std::vector<float>& y,
    CudaOps& cuda) {
    if (static_cast<int>(x.size()) != mat.cols) {
        throw std::runtime_error("linear_forward size mismatch");
    }
    y.resize(mat.rows);
    cuda.matvec(mat.w.data(), x.data(), y.data(), mat.rows, mat.cols);
}

static void linear_backward(
    Matrix& mat,
    const std::vector<float>& x,
    const std::vector<float>& dy,
    std::vector<float>& dx,
    CudaOps& cuda) {
    if (static_cast<int>(x.size()) != mat.cols || static_cast<int>(dy.size()) != mat.rows) {
        throw std::runtime_error("linear_backward size mismatch");
    }
    cuda.outer_add(mat.g.data(), dy.data(), x.data(), mat.rows, mat.cols, 1.0f);
    dx.resize(mat.cols);
    cuda.matvec_t(mat.w.data(), dy.data(), dx.data(), mat.rows, mat.cols);
}

static float forward_train_sequence(
    const Model& model,
    const std::vector<int>& tokens,
    int n,
    std::vector<StepCache>& steps,
    CudaOps& cuda) {
    const HyperParams& hp = model.hp;
    steps.clear();
    steps.reserve(static_cast<size_t>(n));
    float total_loss = 0.0f;

    for (int pos = 0; pos < n; ++pos) {
        steps.emplace_back(hp, model.vocab_size);
        StepCache& sc = steps.back();
        sc.token_id = tokens[static_cast<size_t>(pos)];
        sc.target_id = tokens[static_cast<size_t>(pos + 1)];

        for (int i = 0; i < hp.n_embd; ++i) {
            sc.x_tokpos[static_cast<size_t>(i)] =
                model.wte.w[static_cast<size_t>(sc.token_id) * hp.n_embd + i] +
                model.wpe.w[static_cast<size_t>(pos) * hp.n_embd + i];
        }
        rmsnorm_forward(sc.x_tokpos, sc.x0, sc.inv_rms0);

        std::vector<float> x = sc.x0;
        for (int li = 0; li < hp.n_layer; ++li) {
            LayerStepCache& lc = sc.layers[static_cast<size_t>(li)];
            const LayerParams& layer = model.layers[static_cast<size_t>(li)];

            lc.x_resid_attn = x;
            rmsnorm_forward(lc.x_resid_attn, lc.x_norm1, lc.inv_rms1);

            linear_forward(layer.attn_wq, lc.x_norm1, lc.q, cuda);
            linear_forward(layer.attn_wk, lc.x_norm1, lc.k, cuda);
            linear_forward(layer.attn_wv, lc.x_norm1, lc.v, cuda);

            std::fill(lc.x_attn.begin(), lc.x_attn.end(), 0.0f);
            for (int h = 0; h < hp.n_head; ++h) {
                int hs = h * hp.head_dim;
                std::vector<float> attn_logits(static_cast<size_t>(pos + 1), 0.0f);
                for (int t = 0; t <= pos; ++t) {
                    const std::vector<float>& k_t =
                        (t == pos) ? lc.k : steps[static_cast<size_t>(t)].layers[static_cast<size_t>(li)].k;
                    float dot = 0.0f;
                    for (int j = 0; j < hp.head_dim; ++j) {
                        dot += lc.q[static_cast<size_t>(hs + j)] * k_t[static_cast<size_t>(hs + j)];
                    }
                    attn_logits[static_cast<size_t>(t)] = dot / std::sqrt(static_cast<float>(hp.head_dim));
                }

                std::vector<float> attn_weights;
                stable_softmax(attn_logits, attn_weights);
                lc.attn_weights[static_cast<size_t>(h)] = attn_weights;

                for (int j = 0; j < hp.head_dim; ++j) {
                    float v_sum = 0.0f;
                    for (int t = 0; t <= pos; ++t) {
                        const std::vector<float>& v_t =
                            (t == pos) ? lc.v : steps[static_cast<size_t>(t)].layers[static_cast<size_t>(li)].v;
                        v_sum += attn_weights[static_cast<size_t>(t)] * v_t[static_cast<size_t>(hs + j)];
                    }
                    lc.x_attn[static_cast<size_t>(hs + j)] = v_sum;
                }
            }

            std::vector<float> wo_out;
            linear_forward(layer.attn_wo, lc.x_attn, wo_out, cuda);
            for (int i = 0; i < hp.n_embd; ++i) {
                lc.x_after_attn[static_cast<size_t>(i)] =
                    wo_out[static_cast<size_t>(i)] + lc.x_resid_attn[static_cast<size_t>(i)];
            }

            rmsnorm_forward(lc.x_after_attn, lc.x_norm2, lc.inv_rms2);
            linear_forward(layer.mlp_fc1, lc.x_norm2, lc.fc1_pre, cuda);
            for (size_t i = 0; i < lc.fc1_pre.size(); ++i) {
                float z = lc.fc1_pre[i];
                lc.fc1_act[i] = z > 0.0f ? z * z : 0.0f;
            }
            std::vector<float> fc2_out;
            linear_forward(layer.mlp_fc2, lc.fc1_act, fc2_out, cuda);
            for (int i = 0; i < hp.n_embd; ++i) {
                lc.x_out[static_cast<size_t>(i)] =
                    fc2_out[static_cast<size_t>(i)] + lc.x_after_attn[static_cast<size_t>(i)];
            }
            x = lc.x_out;
        }

        sc.x_final = x;
        linear_forward(model.wte, sc.x_final, sc.logits, cuda);
        total_loss += cross_entropy_with_probs(sc.logits, sc.target_id, sc.probs);
    }

    return total_loss / static_cast<float>(n);
}

static void backward_train_sequence(
    Model& model,
    const std::vector<StepCache>& steps,
    int n,
    CudaOps& cuda) {
    const HyperParams& hp = model.hp;

    std::vector<std::vector<std::vector<float>>> dK(
        static_cast<size_t>(hp.n_layer),
        std::vector<std::vector<float>>(static_cast<size_t>(n), std::vector<float>(hp.n_embd, 0.0f)));
    std::vector<std::vector<std::vector<float>>> dV(
        static_cast<size_t>(hp.n_layer),
        std::vector<std::vector<float>>(static_cast<size_t>(n), std::vector<float>(hp.n_embd, 0.0f)));

    for (int pos = n - 1; pos >= 0; --pos) {
        const StepCache& sc = steps[static_cast<size_t>(pos)];

        std::vector<float> dlogits;
        cross_entropy_grad(sc.probs, sc.target_id, 1.0f / static_cast<float>(n), dlogits);

        std::vector<float> d_x;
        linear_backward(model.wte, sc.x_final, dlogits, d_x, cuda);

        for (int li = hp.n_layer - 1; li >= 0; --li) {
            LayerParams& layer = model.layers[static_cast<size_t>(li)];
            const LayerStepCache& lc = sc.layers[static_cast<size_t>(li)];

            std::vector<float> d_x_after_attn = d_x;

            std::vector<float> d_fc1_act;
            linear_backward(layer.mlp_fc2, lc.fc1_act, d_x, d_fc1_act, cuda);

            std::vector<float> d_fc1_pre(lc.fc1_pre.size(), 0.0f);
            for (size_t i = 0; i < d_fc1_pre.size(); ++i) {
                float z = lc.fc1_pre[i];
                d_fc1_pre[i] = z > 0.0f ? 2.0f * z * d_fc1_act[i] : 0.0f;
            }

            std::vector<float> d_x_norm2;
            linear_backward(layer.mlp_fc1, lc.x_norm2, d_fc1_pre, d_x_norm2, cuda);

            std::vector<float> d_norm2_in;
            rmsnorm_backward(lc.x_after_attn, lc.inv_rms2, d_x_norm2, d_norm2_in);
            add_inplace(d_x_after_attn, d_norm2_in);

            std::vector<float> d_x_resid_attn = d_x_after_attn;
            std::vector<float> d_x_attn;
            linear_backward(layer.attn_wo, lc.x_attn, d_x_after_attn, d_x_attn, cuda);

            std::vector<float> dq(hp.n_embd, 0.0f);
            float inv_sqrt_head = 1.0f / std::sqrt(static_cast<float>(hp.head_dim));

            for (int h = 0; h < hp.n_head; ++h) {
                int hs = h * hp.head_dim;
                int tlen = pos + 1;

                std::vector<float> dweights(static_cast<size_t>(tlen), 0.0f);
                for (int t = 0; t <= pos; ++t) {
                    const std::vector<float>& v_t =
                        (t == pos) ? lc.v : steps[static_cast<size_t>(t)].layers[static_cast<size_t>(li)].v;
                    float dot = 0.0f;
                    for (int j = 0; j < hp.head_dim; ++j) {
                        dot += d_x_attn[static_cast<size_t>(hs + j)] * v_t[static_cast<size_t>(hs + j)];
                    }
                    dweights[static_cast<size_t>(t)] = dot;

                    float wt = lc.attn_weights[static_cast<size_t>(h)][static_cast<size_t>(t)];
                    for (int j = 0; j < hp.head_dim; ++j) {
                        dV[static_cast<size_t>(li)][static_cast<size_t>(t)][static_cast<size_t>(hs + j)] +=
                            wt * d_x_attn[static_cast<size_t>(hs + j)];
                    }
                }

                float sum_dw_w = 0.0f;
                for (int t = 0; t <= pos; ++t) {
                    float wt = lc.attn_weights[static_cast<size_t>(h)][static_cast<size_t>(t)];
                    sum_dw_w += dweights[static_cast<size_t>(t)] * wt;
                }

                for (int t = 0; t <= pos; ++t) {
                    float wt = lc.attn_weights[static_cast<size_t>(h)][static_cast<size_t>(t)];
                    float dlogit = wt * (dweights[static_cast<size_t>(t)] - sum_dw_w);
                    const std::vector<float>& k_t =
                        (t == pos) ? lc.k : steps[static_cast<size_t>(t)].layers[static_cast<size_t>(li)].k;

                    for (int j = 0; j < hp.head_dim; ++j) {
                        dq[static_cast<size_t>(hs + j)] +=
                            dlogit * k_t[static_cast<size_t>(hs + j)] * inv_sqrt_head;
                        dK[static_cast<size_t>(li)][static_cast<size_t>(t)][static_cast<size_t>(hs + j)] +=
                            dlogit * lc.q[static_cast<size_t>(hs + j)] * inv_sqrt_head;
                    }
                }
            }

            std::vector<float> d_x_norm1(hp.n_embd, 0.0f);
            std::vector<float> d_tmp;
            linear_backward(layer.attn_wq, lc.x_norm1, dq, d_tmp, cuda);
            add_inplace(d_x_norm1, d_tmp);
            linear_backward(
                layer.attn_wk, lc.x_norm1, dK[static_cast<size_t>(li)][static_cast<size_t>(pos)], d_tmp, cuda);
            add_inplace(d_x_norm1, d_tmp);
            linear_backward(
                layer.attn_wv, lc.x_norm1, dV[static_cast<size_t>(li)][static_cast<size_t>(pos)], d_tmp, cuda);
            add_inplace(d_x_norm1, d_tmp);

            std::vector<float> d_norm1_in;
            rmsnorm_backward(lc.x_resid_attn, lc.inv_rms1, d_x_norm1, d_norm1_in);
            d_x = d_x_resid_attn;
            add_inplace(d_x, d_norm1_in);
        }

        std::vector<float> d_tokpos;
        rmsnorm_backward(sc.x_tokpos, sc.inv_rms0, d_x, d_tokpos);
        float* g_tok = model.wte.g.data() + static_cast<size_t>(sc.token_id) * hp.n_embd;
        float* g_pos = model.wpe.g.data() + static_cast<size_t>(pos) * hp.n_embd;
        for (int i = 0; i < hp.n_embd; ++i) {
            g_tok[i] += d_tokpos[static_cast<size_t>(i)];
            g_pos[i] += d_tokpos[static_cast<size_t>(i)];
        }
    }
}

static void forward_token_infer(
    const Model& model,
    int token_id,
    int pos_id,
    std::vector<KVLayerCache>& kv,
    std::vector<float>& logits,
    CudaOps& cuda) {
    const HyperParams& hp = model.hp;

    std::vector<float> x(hp.n_embd, 0.0f);
    for (int i = 0; i < hp.n_embd; ++i) {
        x[static_cast<size_t>(i)] =
            model.wte.w[static_cast<size_t>(token_id) * hp.n_embd + i] +
            model.wpe.w[static_cast<size_t>(pos_id) * hp.n_embd + i];
    }

    float inv_rms0 = 0.0f;
    rmsnorm_forward(x, x, inv_rms0);

    for (int li = 0; li < hp.n_layer; ++li) {
        LayerParams const& layer = model.layers[static_cast<size_t>(li)];
        std::vector<float> x_resid_attn = x;
        std::vector<float> x_norm1;
        float inv_rms1 = 0.0f;
        rmsnorm_forward(x_resid_attn, x_norm1, inv_rms1);

        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        linear_forward(layer.attn_wq, x_norm1, q, cuda);
        linear_forward(layer.attn_wk, x_norm1, k, cuda);
        linear_forward(layer.attn_wv, x_norm1, v, cuda);

        kv[static_cast<size_t>(li)].keys.push_back(k);
        kv[static_cast<size_t>(li)].values.push_back(v);

        std::vector<float> x_attn(hp.n_embd, 0.0f);
        for (int h = 0; h < hp.n_head; ++h) {
            int hs = h * hp.head_dim;
            int tlen = static_cast<int>(kv[static_cast<size_t>(li)].keys.size());
            std::vector<float> attn_logits(static_cast<size_t>(tlen), 0.0f);
            for (int t = 0; t < tlen; ++t) {
                const std::vector<float>& kt = kv[static_cast<size_t>(li)].keys[static_cast<size_t>(t)];
                float dot = 0.0f;
                for (int j = 0; j < hp.head_dim; ++j) {
                    dot += q[static_cast<size_t>(hs + j)] * kt[static_cast<size_t>(hs + j)];
                }
                attn_logits[static_cast<size_t>(t)] = dot / std::sqrt(static_cast<float>(hp.head_dim));
            }
            std::vector<float> attn_weights;
            stable_softmax(attn_logits, attn_weights);

            for (int j = 0; j < hp.head_dim; ++j) {
                float sum_v = 0.0f;
                for (int t = 0; t < tlen; ++t) {
                    const std::vector<float>& vt = kv[static_cast<size_t>(li)].values[static_cast<size_t>(t)];
                    sum_v += attn_weights[static_cast<size_t>(t)] * vt[static_cast<size_t>(hs + j)];
                }
                x_attn[static_cast<size_t>(hs + j)] = sum_v;
            }
        }

        std::vector<float> wo_out;
        linear_forward(layer.attn_wo, x_attn, wo_out, cuda);
        for (int i = 0; i < hp.n_embd; ++i) {
            x[static_cast<size_t>(i)] = wo_out[static_cast<size_t>(i)] + x_resid_attn[static_cast<size_t>(i)];
        }

        std::vector<float> x_resid_mlp = x;
        std::vector<float> x_norm2;
        float inv_rms2 = 0.0f;
        rmsnorm_forward(x_resid_mlp, x_norm2, inv_rms2);
        std::vector<float> fc1_pre;
        std::vector<float> fc1_act;
        linear_forward(layer.mlp_fc1, x_norm2, fc1_pre, cuda);
        fc1_act.resize(fc1_pre.size());
        for (size_t i = 0; i < fc1_pre.size(); ++i) {
            float z = fc1_pre[i];
            fc1_act[i] = z > 0.0f ? z * z : 0.0f;
        }
        std::vector<float> fc2_out;
        linear_forward(layer.mlp_fc2, fc1_act, fc2_out, cuda);
        for (int i = 0; i < hp.n_embd; ++i) {
            x[static_cast<size_t>(i)] = fc2_out[static_cast<size_t>(i)] + x_resid_mlp[static_cast<size_t>(i)];
        }
    }

    linear_forward(model.wte, x, logits, cuda);
}

static float grad_norm(const Model& model) {
    double sum_sq = 0.0;
    model.for_each_matrix([&](const Matrix& mat) {
        for (float g : mat.g) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
        }
    });
    return std::sqrt(sum_sq);
}

static void adamw_update(
    Model& model,
    CudaOps& cuda,
    const TrainConfig& cfg,
    float lr_t,
    float b1_prod,
    float b2_prod,
    float grad_scale) {
    float one_minus_b1_prod = 1.0f - b1_prod;
    float one_minus_b2_prod = 1.0f - b2_prod;
    model.for_each_matrix([&](Matrix& mat) {
        cuda.adamw_update(
            mat.w.data(),
            mat.g.data(),
            mat.m.data(),
            mat.v.data(),
            static_cast<int>(mat.w.size()),
            lr_t,
            cfg.beta1,
            cfg.beta2,
            cfg.eps_adam,
            cfg.weight_decay,
            one_minus_b1_prod,
            one_minus_b2_prod,
            grad_scale);
    });
}

static float evaluate_validation(const Model& model, const DataBundle& data, const TrainConfig& cfg, CudaOps& cuda) {
    if (data.val_docs.empty()) {
        return std::nanf("");
    }

    float val_loss = 0.0f;
    int val_n = 0;
    int n_docs = std::min(cfg.val_docs, static_cast<int>(data.val_docs.size()));

    for (int i = 0; i < n_docs; ++i) {
        const std::string& doc = data.val_docs[static_cast<size_t>(i)];
        std::vector<int> tokens = encode_doc(doc, data.stoi, data.bos);
        int vn = std::min(model.hp.block_size, static_cast<int>(tokens.size()) - 1);
        if (vn <= 0) {
            continue;
        }
        std::vector<KVLayerCache> kv(static_cast<size_t>(model.hp.n_layer));
        for (int pos = 0; pos < vn; ++pos) {
            std::vector<float> logits;
            forward_token_infer(
                model,
                tokens[static_cast<size_t>(pos)],
                pos,
                kv,
                logits,
                cuda);
            int target = tokens[static_cast<size_t>(pos + 1)];

            float mx = *std::max_element(logits.begin(), logits.end());
            double sum_exp = 0.0;
            for (float l : logits) {
                sum_exp += std::exp(l - mx);
            }
            val_loss -= logits[static_cast<size_t>(target)] - mx - std::log(sum_exp);
            ++val_n;
        }
    }

    if (val_n == 0) {
        return std::nanf("");
    }
    return val_loss / static_cast<float>(val_n);
}

static int sample_top_k(
    const std::vector<float>& logits,
    int top_k,
    float temperature,
    std::mt19937& rng) {
    int vocab = static_cast<int>(logits.size());
    int k = std::max(1, std::min(top_k, vocab));
    std::vector<int> ids(static_cast<size_t>(vocab));
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(
        ids.begin(),
        ids.begin() + k,
        ids.end(),
        [&](int a, int b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });

    std::vector<float> scaled(static_cast<size_t>(k), 0.0f);
    for (int i = 0; i < k; ++i) {
        scaled[static_cast<size_t>(i)] = logits[static_cast<size_t>(ids[static_cast<size_t>(i)])] / temperature;
    }
    float mx = *std::max_element(scaled.begin(), scaled.end());
    std::vector<float> probs(static_cast<size_t>(k), 0.0f);
    for (int i = 0; i < k; ++i) {
        probs[static_cast<size_t>(i)] = std::exp(scaled[static_cast<size_t>(i)] - mx);
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return ids[static_cast<size_t>(dist(rng))];
}

static TrainConfig parse_args(int argc, char** argv) {
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto read_int = [&](const char* name) -> int {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::stoi(argv[++i]);
        };
        auto read_float = [&](const char* name) -> float {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::stof(argv[++i]);
        };

        if (arg == "--steps") {
            cfg.num_steps = read_int("--steps");
        } else if (arg == "--val-every") {
            cfg.val_every = read_int("--val-every");
        } else if (arg == "--val-docs") {
            cfg.val_docs = read_int("--val-docs");
        } else if (arg == "--samples") {
            cfg.num_samples = read_int("--samples");
        } else if (arg == "--top-k") {
            cfg.top_k = read_int("--top-k");
        } else if (arg == "--temperature") {
            cfg.temperature = read_float("--temperature");
        } else if (arg == "--seed") {
            cfg.seed = read_int("--seed");
        } else if (arg == "--help") {
            std::cout
                << "Usage: microgpt_cuda [options]\n"
                << "  --steps <int>         training steps (default 500)\n"
                << "  --val-every <int>     validation interval (default 100)\n"
                << "  --val-docs <int>      max validation docs per eval (default 20)\n"
                << "  --samples <int>       number of generated samples (default 20)\n"
                << "  --top-k <int>         top-k sampling (default 5)\n"
                << "  --temperature <float> sampling temperature (default 0.6)\n"
                << "  --seed <int>          RNG seed (default 42)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return cfg;
}

int main(int argc, char** argv) {
    try {
        TrainConfig cfg = parse_args(argc, argv);
        HyperParams hp;
        hp.head_dim = hp.n_embd / hp.n_head;
        if (hp.n_embd % hp.n_head != 0) {
            throw std::runtime_error("n_embd must be divisible by n_head");
        }

        std::mt19937 rng(cfg.seed);
        DataBundle data = load_data(rng);
        if (data.train_docs.empty()) {
            throw std::runtime_error("no training documents after split");
        }

        Model model(static_cast<int>(data.itos.size()), hp, rng);
        CudaOps cuda;

        std::cout << "num docs: " << data.docs.size()
                  << " (train: " << data.train_docs.size()
                  << ", val: " << data.val_docs.size() << ")\n";
        std::cout << "vocab size: " << data.itos.size() << "\n";
        std::cout << "num params: " << model.num_params() << "\n";

        float b1_prod = 1.0f;
        float b2_prod = 1.0f;
        constexpr float kPi = 3.14159265358979323846f;

        for (int step = 0; step < cfg.num_steps; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            const std::string& doc =
                data.train_docs[static_cast<size_t>(step % static_cast<int>(data.train_docs.size()))];
            std::vector<int> tokens = encode_doc(doc, data.stoi, data.bos);
            int n = std::min(hp.block_size, static_cast<int>(tokens.size()) - 1);
            if (n <= 0) {
                continue;
            }

            model.zero_grads();
            std::vector<StepCache> steps;
            float loss = forward_train_sequence(model, tokens, n, steps, cuda);
            backward_train_sequence(model, steps, n, cuda);

            float gnorm = grad_norm(model);
            float grad_scale = 1.0f;
            if (gnorm > cfg.max_grad_norm) {
                grad_scale = cfg.max_grad_norm / gnorm;
            }

            float lr_t = cfg.learning_rate * 0.5f * (1.0f + std::cos(kPi * step / cfg.num_steps));
            b1_prod *= cfg.beta1;
            b2_prod *= cfg.beta2;
            adamw_update(model, cuda, cfg, lr_t, b1_prod, b2_prod, grad_scale);

            auto t1 = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "step " << std::setw(4) << (step + 1)
                      << "/" << std::setw(4) << cfg.num_steps
                      << " | loss " << std::fixed << std::setprecision(4) << loss
                      << " | " << ms << "ms\n";

            if ((step + 1) % cfg.val_every == 0) {
                float val = evaluate_validation(model, data, cfg, cuda);
                if (std::isnan(val)) {
                    std::cout << "  val loss: n/a\n";
                } else {
                    std::cout << "  val loss: " << std::fixed << std::setprecision(4) << val << "\n";
                }
            }
        }

        std::cout << "\n--- inference ---\n";
        for (int sample_idx = 0; sample_idx < cfg.num_samples; ++sample_idx) {
            std::vector<KVLayerCache> kv(static_cast<size_t>(hp.n_layer));
            int token_id = data.bos;
            std::cout << "sample " << std::setw(2) << (sample_idx + 1) << ": ";
            for (int pos = 0; pos < hp.block_size; ++pos) {
                std::vector<float> logits;
                forward_token_infer(model, token_id, pos, kv, logits, cuda);
                token_id = sample_top_k(logits, cfg.top_k, cfg.temperature, rng);
                if (token_id == data.bos) {
                    break;
                }
                std::cout << data.itos[static_cast<size_t>(token_id)];
            }
            std::cout << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
