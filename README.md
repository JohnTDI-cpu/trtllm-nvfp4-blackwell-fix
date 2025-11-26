# NVFP4 on RTX 5090: The "Impossible" Guide
### Running Qwen 2.5/3 30B MoE in 4-bit on 32GB VRAM (Consumer Hardware)

**Authors:** Janusz & AI Assistant
**Date:** 2025-11-26
**Tested Configuration:**
*   **GPU:** NVIDIA RTX 5090 (32GB)
*   **OS:** Linux (Kernel 6.14)
*   **Driver:** 580.xx (CUDA 13.0)
*   **Software:** TensorRT-LLM `v0.16.0` / `v1.2.0rc4` (Docker: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4`)
    *   Base Repo: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

> **⚠️ WARNING:** This tutorial involves **patching C++ source code** of the TensorRT-LLM runtime to bypass strict type checks and intentionally leak memory (managed weights workaround). Use at your own risk. This is a dev-environment hotfix.

---

## 1. Performance & Benchmarks

Testowane na **NVIDIA RTX 5090 (32GB)**. Pomiary wykonano na działającym serwerze API (end-to-end).

| Metric | This Setup (RTX 5090) | Cloud API (GPT-4) | Local llama.cpp Q4 (CPU/Mixed) |
| :--- | :--- | :--- | :--- |
| **Throughput** | **~135 tokens/s** 🚀 | ~40 tokens/s | ~20 tokens/s |
| **TTFT** | **~15 ms** ⚡ | 200-500 ms | 50-100 ms |
| **VRAM Usage** | **24.1 GB** | N/A | Varies |
| **Cost/Month** | **$0** (Hardware Owned) | $500-2000+ | $0 |

### 🧠 Context & Architecture
*   **Model:** Qwen 3 30B A3B Instruct (MoE).
*   **Why so fast?** Qwen 3 is a Mixture-of-Experts model. While it has **30B total parameters**, only **~2.4B parameters are active** per token generation.
*   **NVFP4 Impact:** Combined with Blackwell's native NVFP4 tensor cores, memory bandwidth usage is slashed by ~50% vs FP8, allowing the card to hit extremely high token rates usually reserved for 7B models.
*   **Comparison:** Dense 30B models in FP8 typically hit 60-80 t/s. This setup is **~2x faster**.

---

## 2. Prerequisites (The "SWAP" Trick)

The conversion process requires huge amounts of RAM (>100GB). If you have "only" 64GB RAM, you need a massive SWAP file.

```bash
# Create 64GB SWAP file
sudo fallocate -l 64G /swapfile_tmp
sudo chmod 600 /swapfile_tmp
sudo mkswap /swapfile_tmp
sudo swapon /swapfile_tmp
# Verify
free -h
```

---

## 3. Step-by-Step Guide

### Step 1: Quantization (ModelOpt)
*(Script location: `examples/quantization/quantize.py` inside the official TRT-LLM repo/container)*

Use `--device_map cpu` to force loading weights into RAM/SWAP, avoiding "Meta Tensor" errors.

```bash
python3 examples/quantization/quantize.py \
    --model_dir /workspace/models/Qwen3-30B-A3B-Instruct \
    --dtype bfloat16 \
    --qformat nvfp4 \
    --device_map cpu \
    --output_dir /workspace/models/Qwen3-30B-NVFP4-Ckpt
```

### Step 2: Build Engine
Use `--fast_build` and environment variables to reduce compiler VRAM usage.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trtllm-build \
    --checkpoint_dir /workspace/models/Qwen3-30B-NVFP4-Ckpt \
    --gemm_plugin nvfp4 \
    --max_batch_size 1 \
    --max_seq_len 4096 \
    --fast_build \
    --output_dir /workspace/models/Qwen3-30B-NVFP4-Engine
```

### Step 3: The C++ Runtime Patch (CRITICAL)
TRT-LLM v1.2.0rc4 runtime rejects NVFP4 weights loaded via "Managed Weights" because of a type mismatch (`INT8` container vs `FP4` engine expectation) and has a bug in allocation size.

**File:** `cpp/tensorrt_llm/runtime/tllmRuntime.cpp`
**Action:** Apply the following patch to `tllmRuntime.cpp` (around line 820, inside `mManagedWeightsMap.insert` loop):

```cpp
// --- PATCH START ---
#include <cuda_runtime_api.h> // Add at top of file

// Inside setInputTensorsImpl, replace the weight allocation logic:

// Check if Engine expects FP4 (10) but File provides INT8 (2) or UINT8 (5)
if (static_cast<int>(engineDtype) == 10 && (static_cast<int>(trtDtype) == 5 || static_cast<int>(trtDtype) == 2)) 
{
    TLLM_LOG_WARNING("Patching shape/dtype for NVFP4 weight (LEAK MODE): %s", name.c_str());
    
    // Calculate size
    size_t sizeBytes = 1;
    for(int i=0; i<trtDims.nbDims; ++i) sizeBytes *= trtDims.d[i];

    // MANUAL ALLOCATION (Intentional Leak)
    void* rawPtr = nullptr;
    cudaMalloc(&rawPtr, sizeBytes); 
    cudaMemcpy(rawPtr, weight->data(), sizeBytes, cudaMemcpyHostToDevice);

    // Fix Dimensions (Unpack 4-bit)
    auto engineDims = trtDims;
    engineDims.d[engineDims.nbDims - 1] *= 2; 
    
    size_t capacity = 1;
    for(int i=0; i<engineDims.nbDims; ++i) capacity *= engineDims.d[i];
    
    // Wrap existing pointer
    weightsDevice = ITensor::wrap(rawPtr, engineDtype, engineDims, capacity);
}
else {
    // Original logic for other types
    weightsDevice = std::shared_ptr<ITensor>{manager.allocate(MemoryType::kGPU, trtDims, trtDtype)};
    manager.copy(weight->data(), *weightsDevice, MemoryType::kCPU);
}
// --- PATCH END ---
```

**Rebuild:**
```bash
python3 scripts/build_wheel.py --clean --trt_root /usr/local/tensorrt
pip install ./build/tensorrt_llm-*.whl
```

### Step 4: Launch Server
Correctly map ports and use the patched backend.

```bash
trtllm-serve serve /workspace/models/Qwen3-30B-NVFP4-Engine \
    --tokenizer /workspace/models/Qwen3-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --backend tensorrt \
    --kv_cache_free_gpu_memory_fraction 0.4
```

---

## 4. Verification & Troubleshooting

### Verification
Check logs to confirm the patch is working. You should see the "Patching shape/dtype" warning for every layer.
```bash
docker logs trtllm-qwen-hq | grep "Patching shape/dtype" | head -n 5
```
*Output should look like:*
`[WARNING] Patching shape/dtype for NVFP4 weight (LEAK MODE): transformer.layers.0.mlp.fc.weight`

Test the API:
```bash
curl http://localhost:8000/v1/models
# Output: {"object":"list","data":[{"id":"...","object":"model","created":...}]}
```

### Known Issues
1.  **Open WebUI 400 Bad Request:**
    *   *Cause:* WebUI sends `mirostat`, `repeat_penalty` which TRT-LLM doesn't support.
    *   *Fix:* Patch `openai.py` in WebUI to filter parameters OR disable them in WebUI settings.
2.  **Memory Leak:**
    *   *Cause:* `cudaMalloc` is never freed.
    *   *Workaround:* Restart the Docker container to clear VRAM.

### Rollback
To revert the C++ patch and restore original functionality:
```bash
# If you installed via wheel
pip install --force-reinstall nvidia-tensorrt-llm==0.16.0 # (Or verify version in container)

# Easiest way:
# Just restart the original Docker container without mounting/installing the custom wheel.
```
