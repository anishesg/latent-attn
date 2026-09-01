#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "mla_config.cuh"
#include "naive_mla.cuh"
#include "fused_mla.cuh"

// -------------------------------------------------------------------------
// Utilities
// -------------------------------------------------------------------------

static void fill_random_fp16(__half* dst, int n, float scale = 0.1f) {
    for (int i = 0; i < n; ++i)
        dst[i] = __float2half((float(rand()) / RAND_MAX - 0.5f) * 2.0f * scale);
}

static void fill_random_fp32(float* dst, int n, float scale = 0.1f) {
    for (int i = 0; i < n; ++i)
        dst[i] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
}

// Build absorbed query from raw q_nope and W_UK (host side, fp32 for simplicity)
// q_nope: (num_heads, qk_nope_head_dim)
// W_UK:   (num_heads * qk_nope_head_dim, kv_lora_rank) -- note: each head slice
// q_absorbed: (num_heads, kv_lora_rank)
static void compute_q_absorbed_host(
    const float* q_nope, const __half* W_UK,
    float* q_absorbed,
    int num_heads, int qk_nope_head_dim, int kv_lora_rank)
{
    for (int h = 0; h < num_heads; ++h) {
        for (int c = 0; c < kv_lora_rank; ++c) {
            float acc = 0.0f;
            for (int i = 0; i < qk_nope_head_dim; ++i) {
                float q = q_nope[h * qk_nope_head_dim + i];
                float w = __half2float(W_UK[h * qk_nope_head_dim * kv_lora_rank
                                           + i * kv_lora_rank + c]);
                acc += q * w;
            }
            q_absorbed[h * kv_lora_rank + c] = acc;
        }
    }
}

// -------------------------------------------------------------------------
// Metrics
// -------------------------------------------------------------------------

static float cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

static float max_abs_error(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

static float relative_mae(const float* a, const float* b, int n) {
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < n; ++i) {
        num += fabsf(a[i] - b[i]);
        den += fabsf(a[i]);
    }
    return num / (den + 1e-12f);
}

// -------------------------------------------------------------------------
// Single test case
// -------------------------------------------------------------------------

static bool run_test(const MLAConfig& cfg, int kv_len, const char* label) {
    printf("  [%s] kv_len=%d num_heads=%d d_c=%d d_R=%d\n",
           label, kv_len, cfg.num_heads, cfg.kv_lora_rank, cfg.qk_rope_head_dim);

    int dc = cfg.kv_lora_rank;
    int dR = cfg.qk_rope_head_dim;
    int nh = cfg.num_heads;
    int dn = cfg.qk_nope_head_dim;
    int dv = cfg.v_head_dim;
    int S  = kv_len;
    int entry_stride = dc + dR;

    // Allocate host buffers
    size_t kv_bytes     = (size_t)S * entry_stride * sizeof(__half);
    size_t W_UK_bytes   = (size_t)nh * dn * dc * sizeof(__half);
    size_t W_UV_bytes   = (size_t)nh * dv * dc * sizeof(__half);
    size_t q_nope_bytes = (size_t)nh * dn * sizeof(float);
    size_t q_rope_bytes = (size_t)nh * dR * sizeof(float);
    size_t q_abs_bytes  = (size_t)nh * dc * sizeof(float);
    size_t rope_bytes   = (size_t)S * (dR / 2) * sizeof(float);

    __half* h_kv_cache   = (__half*)malloc(kv_bytes);
    __half* h_W_UK       = (__half*)malloc(W_UK_bytes);
    __half* h_W_UV       = (__half*)malloc(W_UV_bytes);
    float*  h_q_nope     = (float*)malloc(q_nope_bytes);
    float*  h_q_rope     = (float*)malloc(q_rope_bytes);
    float*  h_q_absorbed = (float*)malloc(q_abs_bytes);
    float*  h_rope_cos   = (float*)malloc(rope_bytes);
    float*  h_rope_sin   = (float*)malloc(rope_bytes);

    fill_random_fp16(h_kv_cache, S * entry_stride, 0.05f);
    fill_random_fp16(h_W_UK, nh * dn * dc, 0.02f);
    fill_random_fp16(h_W_UV, nh * dv * dc, 0.02f);
    fill_random_fp32(h_q_nope, nh * dn, 0.1f);
    fill_random_fp32(h_q_rope, nh * dR, 0.1f);

    // Build RoPE tables (standard rotary encoding)
    for (int pos = 0; pos < S; ++pos) {
        for (int i = 0; i < dR / 2; ++i) {
            float theta = (float)pos / powf(10000.0f, 2.0f * i / dR);
            h_rope_cos[pos * (dR / 2) + i] = cosf(theta);
            h_rope_sin[pos * (dR / 2) + i] = sinf(theta);
        }
    }

    // Compute absorbed query on host
    // W_UK layout for naive_decompress: (nh * dn, dc) -- each row is a weight vector
    // For absorption: q_absorbed[h, c] = sum_i q_nope[h, i] * W_UK[h*dn+i, c]
    // But naive_decompress uses W_UK as (nh*dn, dc) for K_nope = c_t @ W_UK^T
    // So K_nope[pos, h, i] = sum_c c_t[pos,c] * W_UK[h*dn+i, c]
    // q_nope[h,i] . K_nope[pos,h,i] = sum_i q_nope[h,i] * sum_c c_t[pos,c]*W_UK[h*dn+i,c]
    //                                = sum_c c_t[pos,c] * (sum_i q_nope[h,i]*W_UK[h*dn+i,c])
    //                                = c_t[pos] . q_absorbed[h]
    // where q_absorbed[h,c] = sum_i q_nope[h,i] * W_UK[h*dn+i, c]
    compute_q_absorbed_host(h_q_nope, h_W_UK, h_q_absorbed, nh, dn, dc);

    // Copy to device
    __half *d_kv, *d_W_UK, *d_W_UV;
    float  *d_q_nope, *d_q_rope, *d_q_absorbed, *d_rope_cos, *d_rope_sin;
    float  *d_out_naive, *d_out_fused;

    CUDA_CHECK(cudaMalloc(&d_kv,        kv_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_UK,      W_UK_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_UV,      W_UV_bytes));
    CUDA_CHECK(cudaMalloc(&d_q_nope,    q_nope_bytes));
    CUDA_CHECK(cudaMalloc(&d_q_rope,    q_rope_bytes));
    CUDA_CHECK(cudaMalloc(&d_q_absorbed, q_abs_bytes));
    CUDA_CHECK(cudaMalloc(&d_rope_cos,  rope_bytes));
    CUDA_CHECK(cudaMalloc(&d_rope_sin,  rope_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_naive, (size_t)nh * dv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_fused, (size_t)nh * dv * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_kv,        h_kv_cache,   kv_bytes,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_UK,      h_W_UK,       W_UK_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_UV,      h_W_UV,       W_UV_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_nope,    h_q_nope,     q_nope_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_rope,    h_q_rope,     q_rope_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_absorbed, h_q_absorbed, q_abs_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rope_cos,  h_rope_cos,   rope_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rope_sin,  h_rope_sin,   rope_bytes,   cudaMemcpyHostToDevice));

    // Separate c_t and k_rope arrays for naive path
    // Naive path expects c_t: (S, dc) and k_rope_packed: (S, dR) separately
    __half *d_c_t, *d_k_rope;
    CUDA_CHECK(cudaMalloc(&d_c_t,    (size_t)S * dc * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k_rope, (size_t)S * dR * sizeof(__half)));

    // Unpack kv_cache into c_t and k_rope
    __half* h_c_t    = (__half*)malloc((size_t)S * dc * sizeof(__half));
    __half* h_k_rope = (__half*)malloc((size_t)S * dR * sizeof(__half));
    for (int pos = 0; pos < S; ++pos) {
        memcpy(h_c_t    + pos * dc, h_kv_cache + pos * entry_stride,      dc * sizeof(__half));
        memcpy(h_k_rope + pos * dR, h_kv_cache + pos * entry_stride + dc, dR * sizeof(__half));
    }
    CUDA_CHECK(cudaMemcpy(d_c_t,    h_c_t,    (size_t)S * dc * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_rope, h_k_rope, (size_t)S * dR * sizeof(__half), cudaMemcpyHostToDevice));

    // Run naive reference
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    NaiveMlaWorkspace ws(S, cfg);
    naive_mla(handle, d_c_t, d_k_rope, d_W_UK, d_W_UV,
              d_q_nope, d_q_rope, d_rope_cos, d_rope_sin,
              d_out_naive, ws, cfg);

    // Run fused kernel
    FusedMLAParams params;
    params.kv_cache    = d_kv;
    params.seq_len     = S;
    params.q_absorbed  = d_q_absorbed;
    params.q_rope      = d_q_rope;
    params.W_UV        = d_W_UV;
    params.rope_cos    = d_rope_cos;
    params.rope_sin    = d_rope_sin;
    params.output      = d_out_fused;
    params.cfg         = &cfg;

    CUDA_CHECK(launch_fused_mla(params));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare outputs
    int out_n = nh * dv;
    float* h_naive = (float*)malloc(out_n * sizeof(float));
    float* h_fused = (float*)malloc(out_n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_naive, d_out_naive, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused, d_out_fused, out_n * sizeof(float), cudaMemcpyDeviceToHost));

    float cos_sim = cosine_similarity(h_naive, h_fused, out_n);
    float mae     = max_abs_error(h_naive, h_fused, out_n);
    float rmae    = relative_mae(h_naive, h_fused, out_n);

    bool pass = (cos_sim > 0.998f);
    printf("    cos_sim=%.6f  max_abs_err=%.6f  rel_mae=%.6f  %s\n",
           cos_sim, mae, rmae, pass ? "PASS" : "FAIL");

    // Cleanup
    free(h_kv_cache); free(h_W_UK); free(h_W_UV); free(h_q_nope); free(h_q_rope);
    free(h_q_absorbed); free(h_rope_cos); free(h_rope_sin);
    free(h_c_t); free(h_k_rope); free(h_naive); free(h_fused);
    cudaFree(d_kv); cudaFree(d_W_UK); cudaFree(d_W_UV);
    cudaFree(d_q_nope); cudaFree(d_q_rope); cudaFree(d_q_absorbed);
    cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    cudaFree(d_out_naive); cudaFree(d_out_fused);
    cudaFree(d_c_t); cudaFree(d_k_rope);
    cublasDestroy(handle);

    return pass;
}

int main() {
    srand(42);

    printf("=== MLA Correctness Tests ===\n\n");

    int all_pass = 1;

    // DeepSeek-V2 config
    MLAConfig cfg_v2 = MLAConfig::deepseek_v2();

    // Sweep kv_len
    for (int kv_len : {512, 2048, 8192}) {
        if (!run_test(cfg_v2, kv_len, "deepseek-v2")) all_pass = 0;
    }

    // Smaller head counts for faster testing
    MLAConfig cfg_small = cfg_v2;
    cfg_small.num_heads = 16;
    printf("\n  (num_heads=16)\n");
    for (int kv_len : {512, 2048}) {
        if (!run_test(cfg_small, kv_len, "small-16h")) all_pass = 0;
    }

    cfg_small.num_heads = 64;
    printf("\n  (num_heads=64)\n");
    for (int kv_len : {512, 2048}) {
        if (!run_test(cfg_small, kv_len, "small-64h")) all_pass = 0;
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
