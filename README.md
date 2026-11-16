# Llama.rs SafeTensors Parser & LLM Runtime

A lightweight, pure Rust LLM runtime and SafeTensors model loader designed to parse, inspect, and load Transformer-based architectures without heavy dependencies.

---

## 🚀 Current Development Status

The project has advanced from basic metadata inspection to **full model weight loading and runtime state initialization**. It currently supports:
- SafeTensors binary header parsing and efficient tensor slice reading.
- Native decoding and conversion of tensor data types (`F32`, `BF16`, and `F16`) to `f32`.
- Automatic configuration parsing from `config.json` or fallback architecture inference directly from tensor shapes.
- Complete representation of multi-layer Transformer decoder weights (including attention projections, optional biases, SwiGLU MLP layers, and RMS normalization).
- Pre-allocated scratchpad memory (`RunState`) featuring KV cache buffers and precomputed Rotary Position Embedding (RoPE) cosine/sine tables.

---

## Project Structure

The project consists of the following components:

- **`Cargo.toml`**: Project configuration and dependencies (`serde`, `serde_json`).
- **`src/main.rs`**: Entry point that scans the local `models/` directory, discovers `.safetensors` files, loads the model architecture and weights, initializes execution buffers, and prints detailed diagnostic summaries.
- **`src/loader.rs`**: Core SafeTensors parser (`SafeTensorsModel` and `TensorInfo`) with precise binary header parsing and type conversion routines (`bf16_to_f32`, `f16_to_f32`).
- **`src/model.rs`**: Transformer architecture definitions, including `ModelConfig`, `LayerWeights`, `TransformerWeights`, `Model`, and the `RunState` execution context with RoPE precomputation.

---

## Features

- **Multi-Format Weight Support**: Safely reads and converts `F32`, `BF16`, and `F16` tensor weights into clean `f32` representations.
- **Flexible Configuration**: Reads HuggingFace `config.json` files or automatically infers model dimensions (vocab size, hidden size, attention heads, key-value heads, head dimension) from safetensors weight shapes.
- **Optimized RunState Allocation**: Pre-allocates activation scratchpads and KV caches to avoid runtime allocations during inference.
- **RoPE Precomputation**: Precomputes precision Rotary Position Embedding frequency tables in double precision (`f64`).
- **Memory Diagnostics**: Reports exact RAM consumption for both model weights and runtime computation buffers.

---

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/) (Edition 2024 or compatible toolchain)

### Setup & Usage

1. Create a `models/` folder in the root directory of the project (next to `src/` and `Cargo.toml`).
2. Place your `.safetensors` model file (and optionally its `config.json`) inside the `models/` directory (or a subdirectory).
3. Run the project using Cargo:

```bash
cargo run
```

---

## Future Roadmap

- [ ] Implement a lightweight Tokenizer (BPE/SentencePiece).
- [ ] Implement core forward-pass operations (RMSNorm, Multi-Head Attention, SwiGLU MLP, and LM Head projection).
- [ ] **Enable interactive text generation and sampling.**
