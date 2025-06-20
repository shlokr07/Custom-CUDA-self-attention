#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <random>

// === CUDA Kernel ===
__global__ void multihead_self_attention_kernel(
    const float* __restrict__ E,  // [B, T, D]
    float* __restrict__ output,   // [B, T, D]
    int B, int T, int D, int num_heads
) {
    extern __shared__ float shared[]; // dynamic shared memory

    int b = blockIdx.x;       // batch index
    int h = blockIdx.y;       // head index
    int t = threadIdx.x;      // token index
    int D_head = D / num_heads;

    // Shared memory layout
    float* Q = shared;                                // D_head
    float* K = Q + D_head;                            // T x D_head
    float* V = K + T * D_head;                        // T x D_head
    float* scores = V + T * D_head;                   // T

    const float* E_b = E + b * T * D;
    float* out_b = output + b * T * D;

    // === Load Q_t ===
    int q_offset = t * D + h * D_head;
    for (int i = 0; i < D_head; ++i)
        Q[i] = E_b[q_offset + i];

    // === Load K and V ===
    for (int i = 0; i < T; ++i) {
        int base = i * D + h * D_head;
        for (int j = 0; j < D_head; ++j) {
            K[i * D_head + j] = E_b[base + j];
            V[i * D_head + j] = E_b[base + j];
        }
    }

    // === Compute QK^T / sqrt(d) ===
    float max_score = -1e9f;
    for (int i = 0; i < T; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < D_head; ++j)
            dot += Q[j] * K[i * D_head + j];
        dot /= sqrtf((float)D_head);
        scores[i] = dot;
        if (dot > max_score) max_score = dot;
    }

    // === Softmax ===
    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    for (int i = 0; i < T; ++i)
        scores[i] /= sum;

    // === Weighted sum: A·V ===
    for (int j = 0; j < D_head; ++j) {
        float val = 0.0f;
        for (int i = 0; i < T; ++i)
            val += scores[i] * V[i * D_head + j];
        out_b[t * D + h * D_head + j] = val;
    }
}

void run_self_attention(const float* h_E, float* h_out, int B, int T, int D, int num_heads) {
    int N = B * T * D;
    size_t bytes = N * sizeof(float);

    float *d_E, *d_out;
    cudaMalloc(&d_E, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_E, h_E, bytes, cudaMemcpyHostToDevice);

    dim3 grid(B, num_heads);
    dim3 block(T);
    int D_head = D / num_heads;
    size_t shared_mem = sizeof(float) * (D_head + T * D_head * 2 + T);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    multihead_self_attention_kernel<<<grid, block, shared_mem>>>(d_E, d_out, B, T, D, num_heads);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Self-attention complete. Output shape: [" << B << ", " << T << ", " << D << "]\n";
    std::cout << "Time taken: " << ms << " ms" << std::endl;

    cudaFree(d_E);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const char* filename = "data/embeddings1k.bin";
    FILE* f = fopen(filename, "rb");
    if (!f) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return 1;
    }

    // Assume T and D same as PyTorch 
    const int T = 32;
    const int D = 300;

    // Get file size to infer B
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    rewind(f);

    size_t total_floats = file_size / sizeof(float);
    int B = total_floats / (T * D);

    if (B * T * D != total_floats) {
        std::cerr << "File size is not divisible by T * D. Bad format?" << std::endl;
        return 1;
    }

    size_t total_elems = B * T * D;
    std::vector<float> embeddings(total_elems);
    std::vector<float> output(total_elems, 0.0f);

    size_t read = fread(embeddings.data(), sizeof(float), total_elems, f);
    fclose(f);

    if (read != total_elems) {
        std::cerr << "File read error: expected " << total_elems << ", got " << read << std::endl;
        return 1;
    }

    std::cout << "Loaded embeddings from file: [" << B << ", " << T << ", " << D << "]\n";

    int num_heads = 1;
    run_self_attention(embeddings.data(), output.data(), B, T, D, num_heads);

    return 0;
}