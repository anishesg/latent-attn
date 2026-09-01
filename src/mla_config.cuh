#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

struct MLAConfig {
    int d_model;
    int kv_lora_rank;       // d_c: compressed KV latent dimension
    int qk_nope_head_dim;   // per-head non-RoPE key/query dim
    int qk_rope_head_dim;   // per-head RoPE key/query dim (d_R)
    int num_heads;
    int v_head_dim;

    // DeepSeek-V2 defaults
    static MLAConfig deepseek_v2() {
        return {5120, 512, 128, 64, 128, 128};
    }

    static MLAConfig deepseek_v3() {
        // V3 uses same MLA config as V2 but with 256 heads
        return {7168, 512, 128, 64, 128, 128};
    }

    // Bytes per KV cache entry: packed [c_t || k_rope_t]
    int bytes_per_kv_entry() const {
        return (kv_lora_rank + qk_rope_head_dim) * sizeof(__half);
    }

    // Total KV cache bytes for a given sequence length
    size_t kv_cache_bytes(int seq_len) const {
        return static_cast<size_t>(seq_len) * bytes_per_kv_entry();
    }

    void validate() const {
        if (kv_lora_rank <= 0 || qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0)
            throw std::invalid_argument("MLA dims must be positive");
        if (num_heads <= 0 || v_head_dim <= 0)
            throw std::invalid_argument("num_heads and v_head_dim must be positive");
    }
};

// Packed per-position KV cache entry: [c_t (kv_lora_rank) | k_rope_t (qk_rope_head_dim)]
// Stored as fp16 for bandwidth efficiency.
struct LatentKVCache {
    __half* data;       // device pointer, shape (max_seq_len, kv_lora_rank + qk_rope_head_dim)
    int max_seq_len;
    int entry_stride;   // kv_lora_rank + qk_rope_head_dim
    int current_len;    // number of filled positions

    // Construct on host; caller allocates device memory
    LatentKVCache(__half* device_ptr, int max_len, const MLAConfig& cfg)
        : data(device_ptr), max_seq_len(max_len),
          entry_stride(cfg.kv_lora_rank + cfg.qk_rope_head_dim),
          current_len(0) {}

    // Append one position from host (blocks until copy completes)
    void append_host(const __half* c_t, const __half* k_rope_t,
                     const MLAConfig& cfg, cudaStream_t stream = 0)
    {
        if (current_len >= max_seq_len)
            throw std::runtime_error("KV cache full");
        __half* dst = data + current_len * entry_stride;
        cudaMemcpyAsync(dst, c_t,
                        cfg.kv_lora_rank * sizeof(__half),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dst + cfg.kv_lora_rank, k_rope_t,
                        cfg.qk_rope_head_dim * sizeof(__half),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        ++current_len;
    }

    // Device accessor: returns pointers into the packed buffer for position pos.
    // Call from device code.
    __device__ void get(int pos, int kv_lora_rank,
                        const __half*& c_t_out, const __half*& k_rope_out) const
    {
        const __half* base = data + pos * entry_stride;
        c_t_out   = base;
        k_rope_out = base + kv_lora_rank;
    }

    // Convenience: pointer to packed row pos
    __device__ __host__ const __half* row(int pos) const {
        return data + pos * entry_stride;
    }

    __device__ __host__ __half* row(int pos) {
        return data + pos * entry_stride;
    }
};
