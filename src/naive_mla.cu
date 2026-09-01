#include "naive_mla.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// -------------------------------------------------------------------------
// Workspace management
// -------------------------------------------------------------------------

NaiveMlaWorkspace::NaiveMlaWorkspace(int sl, const MLAConfig& c)
    : seq_len(sl), cfg(c)
{
    CUDA_CHECK(cudaMalloc(&K_nope, (size_t)sl * c.num_heads * c.qk_nope_head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&K_rope, (size_t)sl * c.qk_rope_head_dim               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&V,      (size_t)sl * c.num_heads * c.v_head_dim        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, (size_t)c.num_heads * c.v_head_dim             * sizeof(float)));
}

NaiveMlaWorkspace::~NaiveMlaWorkspace() {
    cudaFree(K_nope);
    cudaFree(K_rope);
    cudaFree(V);
    cudaFree(output);
}

// -------------------------------------------------------------------------
// RoPE kernel: rotate k_rope in-place
// Input k: (seq_len, d_R), cos/sin: (max_seq, d_R/2), positions 0..seq_len-1
// -------------------------------------------------------------------------
__global__ void rope_rotate_kernel(
    float* k,               // (seq_len, d_R) in-place, fp32
    const float* cos_tab,   // (max_seq, d_R/2)
    const float* sin_tab,   // (max_seq, d_R/2)
    int seq_len, int d_R, int max_seq)
{
    int pos  = blockIdx.x;
    int half = d_R / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float x0 = k[pos * d_R + i];
        float x1 = k[pos * d_R + i + half];
        float c  = cos_tab[pos * half + i];
        float s  = sin_tab[pos * half + i];
        k[pos * d_R + i]        = x0 * c - x1 * s;
        k[pos * d_R + i + half] = x0 * s + x1 * c;
    }
}

// Convert packed fp16 k_rope to fp32 and apply RoPE
__global__ void copy_rope_fp16_to_fp32(
    const __half* src, float* dst, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = __half2float(src[i]);
}

// -------------------------------------------------------------------------
// Decompress K and V from latent cache
// -------------------------------------------------------------------------
void naive_decompress_kv(
    cublasHandle_t handle,
    const __half* c_t,
    const __half* k_rope_packed,
    const __half* W_UK,
    const __half* W_UV,
    const float*  rope_cos,
    const float*  rope_sin,
    NaiveMlaWorkspace& ws,
    const MLAConfig& cfg,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int S  = ws.seq_len;
    int dc = cfg.kv_lora_rank;
    int nh = cfg.num_heads;
    int dn = cfg.qk_nope_head_dim;
    int dv = cfg.v_head_dim;
    int dR = cfg.qk_rope_head_dim;

    // cuBLAS uses column-major. Our matrices are row-major (C-order).
    // For row-major A (m x k) and B (k x n) -> C (m x n):
    //   cuBLAS sees A^T (k x m) and B^T (n x k), computes B^T * A^T = (A*B)^T
    // So call: gemm(B, A) with transposed roles to get C = A * B in row-major.

    // K_nope = c_t @ W_UK^T  =>  shape (S, nh*dn)
    // c_t: (S, dc), W_UK: (nh*dn, dc)
    // In column-major: C(nh*dn, S) = W_UK(nh*dn, dc) * c_t^T(dc, S)
    // => gemm: op_A=N(W_UK), op_B=T(c_t), M=nh*dn, N=S, K=dc
    {
        float alpha = 1.0f, beta = 0.0f;
        // Use Sgemm with fp16->fp32 conversion via cublasGemmEx
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            S, nh * dn, dc,
            &alpha,
            c_t,  CUDA_R_16F, dc,
            W_UK, CUDA_R_16F, dc,
            &beta,
            ws.K_nope, CUDA_R_32F, S,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // Result K_nope is column-major (nh*dn, S) which is row-major (S, nh*dn) -- matches our layout
    }

    // V = c_t @ W_UV^T  =>  shape (S, nh*dv)
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            S, nh * dv, dc,
            &alpha,
            c_t,  CUDA_R_16F, dc,
            W_UV, CUDA_R_16F, dc,
            &beta,
            ws.V, CUDA_R_32F, S,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Copy fp16 k_rope to fp32 workspace
    int total_rope = S * dR;
    int threads = 256;
    int blocks  = (total_rope + threads - 1) / threads;
    copy_rope_fp16_to_fp32<<<blocks, threads, 0, stream>>>(k_rope_packed, ws.K_rope, total_rope);

    // Apply RoPE to each position
    rope_rotate_kernel<<<S, 128, 0, stream>>>(ws.K_rope, rope_cos, rope_sin, S, dR, S);

    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------------------
// Attention kernel: single query token against full decompressed K/V
// Scores for each head: q_nope_h . K_nope[pos, h] + q_rope_h . K_rope_rope[pos]
// -------------------------------------------------------------------------
__global__ void naive_attention_kernel(
    const float* q_nope,    // (num_heads, qk_nope_head_dim)
    const float* q_rope,    // (num_heads, qk_rope_head_dim)
    const float* K_nope,    // (seq_len, num_heads, qk_nope_head_dim) -- row-major
    const float* K_rope,    // (seq_len, qk_rope_head_dim)
    const float* V,         // (seq_len, num_heads, v_head_dim)
    float*       output,    // (num_heads, v_head_dim)
    int seq_len, int num_heads,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim,
    float scale)
{
    int h = blockIdx.x;
    if (h >= num_heads) return;

    extern __shared__ float smem[];
    float* scores = smem;                             // seq_len floats
    float* v_accum = smem + seq_len;                  // v_head_dim floats

    for (int i = threadIdx.x; i < v_head_dim; i += blockDim.x)
        v_accum[i] = 0.0f;

    const float* qn = q_nope + h * qk_nope_head_dim;
    const float* qr = q_rope + h * qk_rope_head_dim;

    // Compute scores
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        const float* kn = K_nope + (pos * num_heads + h) * qk_nope_head_dim;
        const float* kr = K_rope + pos * qk_rope_head_dim;
        float s = 0.0f;
        for (int i = 0; i < qk_nope_head_dim; ++i) s += qn[i] * kn[i];
        for (int i = 0; i < qk_rope_head_dim; ++i) s += qr[i] * kr[i];
        scores[pos] = s * scale;
    }
    __syncthreads();

    // Online softmax (single-threaded for simplicity in reference)
    if (threadIdx.x == 0) {
        float max_s = -1e38f;
        for (int pos = 0; pos < seq_len; ++pos)
            max_s = fmaxf(max_s, scores[pos]);
        float sumexp = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            scores[pos] = expf(scores[pos] - max_s);
            sumexp += scores[pos];
        }
        for (int pos = 0; pos < seq_len; ++pos)
            scores[pos] /= sumexp;
    }
    __syncthreads();

    // Weighted sum of V
    for (int pos = 0; pos < seq_len; ++pos) {
        float w = scores[pos];
        const float* v = V + (pos * num_heads + h) * v_head_dim;
        for (int i = threadIdx.x; i < v_head_dim; i += blockDim.x)
            v_accum[i] += w * v[i];
        __syncthreads();
    }
    __syncthreads();

    float* out = output + h * v_head_dim;
    for (int i = threadIdx.x; i < v_head_dim; i += blockDim.x)
        out[i] = v_accum[i];
}

void naive_attention(
    const float* q_nope,
    const float* q_rope,
    NaiveMlaWorkspace& ws,
    const MLAConfig& cfg,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim));
    int S = ws.seq_len;

    // Shared memory: scores (S floats) + v_accum (v_head_dim floats)
    size_t smem = (S + cfg.v_head_dim) * sizeof(float);

    naive_attention_kernel<<<cfg.num_heads, 128, smem, stream>>>(
        q_nope, q_rope,
        ws.K_nope, ws.K_rope, ws.V,
        ws.output,
        S, cfg.num_heads,
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim,
        scale);

    CUDA_CHECK(cudaGetLastError());
}

void naive_mla(
    cublasHandle_t handle,
    const __half* c_t,
    const __half* k_rope_packed,
    const __half* W_UK,
    const __half* W_UV,
    const float*  q_nope,
    const float*  q_rope,
    const float*  rope_cos,
    const float*  rope_sin,
    float*        out_fp32,
    NaiveMlaWorkspace& ws,
    const MLAConfig& cfg,
    cudaStream_t stream)
{
    naive_decompress_kv(handle, c_t, k_rope_packed, W_UK, W_UV,
                        rope_cos, rope_sin, ws, cfg, stream);
    naive_attention(q_nope, q_rope, ws, cfg, stream);
    CUDA_CHECK(cudaMemcpyAsync(out_fp32, ws.output,
        cfg.num_heads * cfg.v_head_dim * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
