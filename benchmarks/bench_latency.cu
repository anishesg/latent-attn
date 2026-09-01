#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "mla_config.cuh"
#include "naive_mla.cuh"
#include "fused_mla.cuh"

static void fill_random_fp16(__half* dst, int n, float scale = 0.05f) {
    for (int i = 0; i < n; ++i)
        dst[i] = __float2half((float(rand()) / RAND_MAX - 0.5f) * 2.0f * scale);
}

static void fill_random_fp32(float* dst, int n, float scale = 0.1f) {
    for (int i = 0; i < n; ++i)
        dst[i] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
}

// CUDA event timer
struct EventTimer {
    cudaEvent_t start, stop;
    EventTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~EventTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    float end(cudaStream_t s = 0)  {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

struct BenchResult {
    float naive_decomp_ms;
    float naive_attn_ms;
    float naive_total_ms;
    float fused_ms;
    size_t naive_bytes_read;  // bytes from global memory per decode step (K+V)
    size_t fused_bytes_read;  // bytes from global memory per decode step (c_t+k_rope)
    float naive_bw_gb;
    float fused_bw_gb;
};

static BenchResult benchmark(const MLAConfig& cfg, int kv_len, int warmup = 3, int iters = 20) {
    int dc = cfg.kv_lora_rank;
    int dR = cfg.qk_rope_head_dim;
    int nh = cfg.num_heads;
    int dn = cfg.qk_nope_head_dim;
    int dv = cfg.v_head_dim;
    int S  = kv_len;
    int entry_stride = dc + dR;

    // Allocate and initialize all buffers
    __half *d_kv, *d_W_UK, *d_W_UV, *d_c_t, *d_k_rope;
    float  *d_q_nope, *d_q_rope, *d_q_absorbed, *d_rope_cos, *d_rope_sin;
    float  *d_out_naive, *d_out_fused;

    CUDA_CHECK(cudaMalloc(&d_kv,        (size_t)S * entry_stride * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_c_t,       (size_t)S * dc * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k_rope,    (size_t)S * dR * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_W_UK,      (size_t)nh * dn * dc * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_W_UV,      (size_t)nh * dv * dc * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_q_nope,    (size_t)nh * dn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_rope,    (size_t)nh * dR * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_absorbed,(size_t)nh * dc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rope_cos,  (size_t)S * (dR/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rope_sin,  (size_t)S * (dR/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_naive, (size_t)nh * dv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_fused, (size_t)nh * dv * sizeof(float)));

    // Initialize with random data
    {
        size_t n_kv = S * entry_stride;
        __half* h = (__half*)malloc(n_kv * sizeof(__half));
        fill_random_fp16(h, n_kv);
        CUDA_CHECK(cudaMemcpy(d_kv, h, n_kv * sizeof(__half), cudaMemcpyHostToDevice));

        // Unpack c_t and k_rope
        __half* h_ct = (__half*)malloc(S * dc * sizeof(__half));
        __half* h_kr = (__half*)malloc(S * dR * sizeof(__half));
        for (int pos = 0; pos < S; ++pos) {
            memcpy(h_ct + pos*dc, h + pos*entry_stride,      dc*sizeof(__half));
            memcpy(h_kr + pos*dR, h + pos*entry_stride + dc, dR*sizeof(__half));
        }
        CUDA_CHECK(cudaMemcpy(d_c_t,    h_ct, S*dc*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_rope, h_kr, S*dR*sizeof(__half), cudaMemcpyHostToDevice));
        free(h); free(h_ct); free(h_kr);
    }
    {
        __half* h = (__half*)malloc(nh * dn * dc * sizeof(__half));
        fill_random_fp16(h, nh * dn * dc, 0.02f);
        CUDA_CHECK(cudaMemcpy(d_W_UK, h, nh*dn*dc*sizeof(__half), cudaMemcpyHostToDevice));
        free(h);
    }
    {
        __half* h = (__half*)malloc(nh * dv * dc * sizeof(__half));
        fill_random_fp16(h, nh * dv * dc, 0.02f);
        CUDA_CHECK(cudaMemcpy(d_W_UV, h, nh*dv*dc*sizeof(__half), cudaMemcpyHostToDevice));
        free(h);
    }
    {
        float* h = (float*)malloc(nh * dn * sizeof(float));
        fill_random_fp32(h, nh * dn);
        CUDA_CHECK(cudaMemcpy(d_q_nope, h, nh*dn*sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }
    {
        float* h = (float*)malloc(nh * dR * sizeof(float));
        fill_random_fp32(h, nh * dR);
        CUDA_CHECK(cudaMemcpy(d_q_rope, h, nh*dR*sizeof(float), cudaMemcpyHostToDevice));
        // Use same for absorbed (approximate)
        float* hab = (float*)malloc(nh * dc * sizeof(float));
        fill_random_fp32(hab, nh * dc, 0.02f);
        CUDA_CHECK(cudaMemcpy(d_q_absorbed, hab, nh*dc*sizeof(float), cudaMemcpyHostToDevice));
        free(h); free(hab);
    }
    {
        float* h = (float*)malloc(S * (dR/2) * sizeof(float));
        for (int pos = 0; pos < S; ++pos)
            for (int i = 0; i < dR/2; ++i) {
                float theta = (float)pos / powf(10000.0f, 2.0f*i/dR);
                h[pos*(dR/2)+i] = cosf(theta);
            }
        CUDA_CHECK(cudaMemcpy(d_rope_cos, h, S*(dR/2)*sizeof(float), cudaMemcpyHostToDevice));
        for (int pos = 0; pos < S; ++pos)
            for (int i = 0; i < dR/2; ++i) {
                float theta = (float)pos / powf(10000.0f, 2.0f*i/dR);
                h[pos*(dR/2)+i] = sinf(theta);
            }
        CUDA_CHECK(cudaMemcpy(d_rope_sin, h, S*(dR/2)*sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    NaiveMlaWorkspace ws(S, cfg);

    FusedMLAParams fp;
    fp.kv_cache   = d_kv;
    fp.seq_len    = S;
    fp.q_absorbed = d_q_absorbed;
    fp.q_rope     = d_q_rope;
    fp.W_UV       = d_W_UV;
    fp.rope_cos   = d_rope_cos;
    fp.rope_sin   = d_rope_sin;
    fp.output     = d_out_fused;
    fp.cfg        = &cfg;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        naive_mla(handle, d_c_t, d_k_rope, d_W_UK, d_W_UV,
                  d_q_nope, d_q_rope, d_rope_cos, d_rope_sin,
                  d_out_naive, ws, cfg);
        launch_fused_mla(fp);
        cudaDeviceSynchronize();
    }

    EventTimer timer;
    float total_naive_decomp = 0, total_naive_attn = 0, total_fused = 0;

    for (int iter = 0; iter < iters; ++iter) {
        // Time naive decompression
        timer.begin();
        naive_decompress_kv(handle, d_c_t, d_k_rope, d_W_UK, d_W_UV,
                            d_rope_cos, d_rope_sin, ws, cfg);
        cudaDeviceSynchronize();
        total_naive_decomp += timer.end();

        // Time naive attention
        timer.begin();
        naive_attention(d_q_nope, d_q_rope, ws, cfg);
        cudaDeviceSynchronize();
        total_naive_attn += timer.end();

        // Time fused
        timer.begin();
        launch_fused_mla(fp);
        cudaDeviceSynchronize();
        total_fused += timer.end();
    }

    // Memory bandwidth analysis
    // Naive: decompression reads c_t (S*dc) + writes K_nope (S*nh*dn) + writes V (S*nh*dv)
    //        attention reads K_nope + K_rope + V
    size_t naive_decomp_read  = (size_t)S * dc * sizeof(__half)
                              + (size_t)S * dR * sizeof(__half);
    size_t naive_decomp_write = (size_t)S * nh * dn * sizeof(float)
                              + (size_t)S * nh * dv * sizeof(float)
                              + (size_t)S * dR * sizeof(float);
    size_t naive_attn_read    = (size_t)S * nh * dn * sizeof(float)
                              + (size_t)S * dR * sizeof(float)
                              + (size_t)S * nh * dv * sizeof(float);
    size_t naive_total_bytes  = naive_decomp_read + naive_decomp_write + naive_attn_read;

    // Fused: reads kv_cache once (S * (dc+dR)) per pass, but does 2 passes (find max, accumulate)
    size_t fused_bytes = 2 * (size_t)S * entry_stride * sizeof(__half);

    float naive_ms   = (total_naive_decomp + total_naive_attn) / iters;
    float decomp_ms  = total_naive_decomp / iters;
    float attn_ms    = total_naive_attn / iters;
    float fused_ms   = total_fused / iters;

    float naive_bw = (float)naive_total_bytes / (naive_ms * 1e6f);  // GB/s
    float fused_bw = (float)fused_bytes       / (fused_ms  * 1e6f);  // GB/s

    BenchResult res;
    res.naive_decomp_ms  = decomp_ms;
    res.naive_attn_ms    = attn_ms;
    res.naive_total_ms   = naive_ms;
    res.fused_ms         = fused_ms;
    res.naive_bytes_read = naive_total_bytes;
    res.fused_bytes_read = fused_bytes;
    res.naive_bw_gb      = naive_bw;
    res.fused_bw_gb      = fused_bw;

    // Cleanup
    cudaFree(d_kv); cudaFree(d_c_t); cudaFree(d_k_rope);
    cudaFree(d_W_UK); cudaFree(d_W_UV);
    cudaFree(d_q_nope); cudaFree(d_q_rope); cudaFree(d_q_absorbed);
    cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    cudaFree(d_out_naive); cudaFree(d_out_fused);
    cublasDestroy(handle);

    return res;
}

static void print_header() {
    printf("%-12s %-8s %-8s | %-10s %-10s %-10s | %-10s | %-8s %-8s | %-8s\n",
           "Config", "heads", "kv_len",
           "decomp_ms", "attn_ms", "naive_ms",
           "fused_ms",
           "speedup", "naive_bw", "fused_bw");
    printf("%s\n", std::string(110, '-').c_str());
}

int main() {
    srand(42);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s  Peak BW: %.1f GB/s\n\n", prop.name,
           2.0f * prop.memoryClockRate * 1e3f * prop.memoryBusWidth / 8.0f / 1e9f);

    print_header();

    MLAConfig cfgs[] = {MLAConfig::deepseek_v2(), MLAConfig::deepseek_v3()};
    const char* cfg_names[] = {"dsv2", "dsv3"};
    int kv_lens[] = {512, 2048, 8192, 32768};

    for (int ci = 0; ci < 2; ++ci) {
        for (int kv_len : kv_lens) {
            MLAConfig& cfg = cfgs[ci];
            BenchResult r = benchmark(cfg, kv_len);

            float speedup = r.naive_total_ms / r.fused_ms;
            float tokens_per_sec_naive = 1000.0f / r.naive_total_ms;
            float tokens_per_sec_fused = 1000.0f / r.fused_ms;

            printf("%-12s %-8d %-8d | %-10.3f %-10.3f %-10.3f | %-10.3f | %-8.2fx %-8.1f | %-8.1f\n",
                   cfg_names[ci], cfg.num_heads, kv_len,
                   r.naive_decomp_ms, r.naive_attn_ms, r.naive_total_ms,
                   r.fused_ms,
                   speedup, r.naive_bw_gb, r.fused_bw_gb);
        }
    }

    printf("\n--- Decode throughput (tokens/sec, single token decode) ---\n");
    for (int ci = 0; ci < 2; ++ci) {
        for (int kv_len : {512, 8192}) {
            MLAConfig& cfg = cfgs[ci];
            BenchResult r = benchmark(cfg, kv_len, 2, 10);
            printf("  %s kv=%6d  naive=%.0f tok/s  fused=%.0f tok/s  fused KV bytes=%zu KB\n",
                   cfg_names[ci], kv_len,
                   1000.0f / r.naive_total_ms, 1000.0f / r.fused_ms,
                   r.fused_bytes_read / 1024);
        }
    }

    return 0;
}
