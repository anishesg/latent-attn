#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "mla_config.cuh"
#include "naive_mla.cuh"
#include "fused_mla.cuh"
#include "fused_mla_batched.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static MLAConfig config_from_args(int d_model, int kv_lora_rank, int qk_nope_head_dim,
                                   int qk_rope_head_dim, int num_heads, int v_head_dim) {
    MLAConfig cfg;
    cfg.d_model           = d_model;
    cfg.kv_lora_rank      = kv_lora_rank;
    cfg.qk_nope_head_dim  = qk_nope_head_dim;
    cfg.qk_rope_head_dim  = qk_rope_head_dim;
    cfg.num_heads         = num_heads;
    cfg.v_head_dim        = v_head_dim;
    cfg.validate();
    return cfg;
}

static void check_fp16_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dtype() == torch::kFloat16,
                name, " must be float16, got ", t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
}

static void check_fp32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dtype() == torch::kFloat32,
                name, " must be float32, got ", t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
}

// ---------------------------------------------------------------------------
// naive_mla_forward
//   kv_cache:   (seq_len, kv_lora_rank + qk_rope_head_dim) fp16  [packed]
//   W_UK:       (num_heads * qk_nope_head_dim, kv_lora_rank) fp16
//   W_UV:       (num_heads * v_head_dim,       kv_lora_rank) fp16
//   q_nope:     (num_heads, qk_nope_head_dim) fp32
//   q_rope:     (num_heads, qk_rope_head_dim) fp32
//   rope_cos:   (seq_len, qk_rope_head_dim/2) fp32
//   rope_sin:   (seq_len, qk_rope_head_dim/2) fp32
//   returns:    (num_heads, v_head_dim) fp32
// ---------------------------------------------------------------------------
torch::Tensor naive_mla_forward(
    torch::Tensor kv_cache,
    torch::Tensor W_UK,
    torch::Tensor W_UV,
    torch::Tensor q_nope,
    torch::Tensor q_rope,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    int d_model, int kv_lora_rank, int qk_nope_head_dim,
    int qk_rope_head_dim, int num_heads, int v_head_dim)
{
    check_fp16_contiguous(kv_cache, "kv_cache");
    check_fp16_contiguous(W_UK, "W_UK");
    check_fp16_contiguous(W_UV, "W_UV");
    check_fp32_contiguous(q_nope, "q_nope");
    check_fp32_contiguous(q_rope, "q_rope");
    check_fp32_contiguous(rope_cos, "rope_cos");
    check_fp32_contiguous(rope_sin, "rope_sin");

    MLAConfig cfg = config_from_args(d_model, kv_lora_rank, qk_nope_head_dim,
                                      qk_rope_head_dim, num_heads, v_head_dim);

    int seq_len     = kv_cache.size(0);
    int entry_stride = kv_lora_rank + qk_rope_head_dim;
    TORCH_CHECK(kv_cache.size(1) == entry_stride,
                "kv_cache dim1 must be ", entry_stride);
    TORCH_CHECK(q_nope.size(0) == num_heads && q_nope.size(1) == qk_nope_head_dim,
                "q_nope shape mismatch");
    TORCH_CHECK(q_rope.size(0) == num_heads && q_rope.size(1) == qk_rope_head_dim,
                "q_rope shape mismatch");

    // Split kv_cache into c_t and k_rope
    auto c_t       = kv_cache.slice(1, 0, kv_lora_rank).contiguous();
    auto k_rope_pk = kv_cache.slice(1, kv_lora_rank, entry_stride).contiguous();

    auto output = torch::empty({num_heads, v_head_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(kv_cache.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    NaiveMlaWorkspace ws(seq_len, cfg);

    naive_mla(
        handle,
        reinterpret_cast<const __half*>(c_t.data_ptr()),
        reinterpret_cast<const __half*>(k_rope_pk.data_ptr()),
        reinterpret_cast<const __half*>(W_UK.data_ptr()),
        reinterpret_cast<const __half*>(W_UV.data_ptr()),
        q_nope.data_ptr<float>(),
        q_rope.data_ptr<float>(),
        rope_cos.data_ptr<float>(),
        rope_sin.data_ptr<float>(),
        output.data_ptr<float>(),
        ws, cfg, stream);

    cublasDestroy(handle);
    return output;
}

// ---------------------------------------------------------------------------
// fused_mla_forward
//   kv_cache:    (seq_len, kv_lora_rank + qk_rope_head_dim) fp16
//   q_absorbed:  (num_heads, kv_lora_rank) fp32
//   q_rope:      (num_heads, qk_rope_head_dim) fp32
//   W_UV:        (num_heads * v_head_dim, kv_lora_rank) fp16
//   rope_cos:    (seq_len, qk_rope_head_dim/2) fp32
//   rope_sin:    (seq_len, qk_rope_head_dim/2) fp32
//   returns:     (num_heads, v_head_dim) fp32
// ---------------------------------------------------------------------------
torch::Tensor fused_mla_forward(
    torch::Tensor kv_cache,
    torch::Tensor q_absorbed,
    torch::Tensor q_rope,
    torch::Tensor W_UV,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    int d_model, int kv_lora_rank, int qk_nope_head_dim,
    int qk_rope_head_dim, int num_heads, int v_head_dim)
{
    check_fp16_contiguous(kv_cache, "kv_cache");
    check_fp32_contiguous(q_absorbed, "q_absorbed");
    check_fp32_contiguous(q_rope, "q_rope");
    check_fp16_contiguous(W_UV, "W_UV");
    check_fp32_contiguous(rope_cos, "rope_cos");
    check_fp32_contiguous(rope_sin, "rope_sin");

    MLAConfig cfg = config_from_args(d_model, kv_lora_rank, qk_nope_head_dim,
                                      qk_rope_head_dim, num_heads, v_head_dim);

    int seq_len = kv_cache.size(0);
    TORCH_CHECK(kv_cache.size(1) == kv_lora_rank + qk_rope_head_dim,
                "kv_cache dim1 mismatch");
    TORCH_CHECK(q_absorbed.size(0) == num_heads && q_absorbed.size(1) == kv_lora_rank,
                "q_absorbed shape mismatch");
    TORCH_CHECK(q_rope.size(0) == num_heads && q_rope.size(1) == qk_rope_head_dim,
                "q_rope shape mismatch");

    auto output = torch::empty({num_heads, v_head_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(kv_cache.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    FusedMLAParams p;
    p.kv_cache   = reinterpret_cast<const __half*>(kv_cache.data_ptr());
    p.seq_len    = seq_len;
    p.q_absorbed = q_absorbed.data_ptr<float>();
    p.q_rope     = q_rope.data_ptr<float>();
    p.W_UV       = reinterpret_cast<const __half*>(W_UV.data_ptr());
    p.rope_cos   = rope_cos.data_ptr<float>();
    p.rope_sin   = rope_sin.data_ptr<float>();
    p.output     = output.data_ptr<float>();
    p.cfg        = &cfg;

    // cfg is stack-local; kernel runs synchronously enough that this is safe
    // (launch_fused_mla does not store the pointer beyond the call)
    cudaError_t err = launch_fused_mla(p, stream);
    TORCH_CHECK(err == cudaSuccess, "launch_fused_mla failed: ", cudaGetErrorString(err));

    return output;
}

// ---------------------------------------------------------------------------
// batched_fused_mla_forward
//   kv_cache:    (seq_len, kv_lora_rank + qk_rope_head_dim) fp16  [shared]
//   q_absorbed:  (batch, num_heads, kv_lora_rank) fp32
//   q_rope:      (batch, num_heads, qk_rope_head_dim) fp32
//   W_UV:        (num_heads * v_head_dim, kv_lora_rank) fp16
//   rope_cos:    (seq_len, qk_rope_head_dim/2) fp32
//   rope_sin:    (seq_len, qk_rope_head_dim/2) fp32
//   returns:     (batch, num_heads, v_head_dim) fp32
// ---------------------------------------------------------------------------
torch::Tensor batched_fused_mla_forward(
    torch::Tensor kv_cache,
    torch::Tensor q_absorbed,
    torch::Tensor q_rope,
    torch::Tensor W_UV,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    int d_model, int kv_lora_rank, int qk_nope_head_dim,
    int qk_rope_head_dim, int num_heads, int v_head_dim)
{
    check_fp16_contiguous(kv_cache, "kv_cache");
    check_fp32_contiguous(q_absorbed, "q_absorbed");
    check_fp32_contiguous(q_rope, "q_rope");
    check_fp16_contiguous(W_UV, "W_UV");
    check_fp32_contiguous(rope_cos, "rope_cos");
    check_fp32_contiguous(rope_sin, "rope_sin");

    MLAConfig cfg = config_from_args(d_model, kv_lora_rank, qk_nope_head_dim,
                                      qk_rope_head_dim, num_heads, v_head_dim);

    TORCH_CHECK(q_absorbed.dim() == 3, "q_absorbed must be 3D (batch, num_heads, kv_lora_rank)");
    int batch    = q_absorbed.size(0);
    int seq_len  = kv_cache.size(0);
    TORCH_CHECK(q_absorbed.size(1) == num_heads && q_absorbed.size(2) == kv_lora_rank,
                "q_absorbed shape mismatch");
    TORCH_CHECK(q_rope.size(0) == batch && q_rope.size(1) == num_heads && q_rope.size(2) == qk_rope_head_dim,
                "q_rope shape mismatch");
    TORCH_CHECK(kv_cache.size(1) == kv_lora_rank + qk_rope_head_dim, "kv_cache dim1 mismatch");

    auto output = torch::empty({batch, num_heads, v_head_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(kv_cache.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    BatchedFusedMLAParams p;
    p.kv_cache   = reinterpret_cast<const __half*>(kv_cache.data_ptr());
    p.seq_len    = seq_len;
    p.batch_size = batch;
    p.q_absorbed = q_absorbed.data_ptr<float>();
    p.q_rope     = q_rope.data_ptr<float>();
    p.W_UV       = reinterpret_cast<const __half*>(W_UV.data_ptr());
    p.rope_cos   = rope_cos.data_ptr<float>();
    p.rope_sin   = rope_sin.data_ptr<float>();
    p.output     = output.data_ptr<float>();
    p.cfg        = &cfg;

    cudaError_t err = launch_batched_fused_mla(p, stream);
    TORCH_CHECK(err == cudaSuccess, "launch_batched_fused_mla failed: ", cudaGetErrorString(err));

    return output;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_mla_forward", &naive_mla_forward,
          "Naive MLA forward (decompress K/V then attend)");
    m.def("fused_mla_forward", &fused_mla_forward,
          "Fused absorbed MLA forward (single token, online softmax in latent space)");
    m.def("batched_fused_mla_forward", &batched_fused_mla_forward,
          "Batched fused MLA forward (batch tokens, shared KV cache)");
}
