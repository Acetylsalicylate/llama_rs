# Llama.rs SafeTensors Parser & Metadata Reader

A lightweight Rust utility designed to scan, parse, and inspect `.safetensors` model files. This project serves as the foundational data-loading layer for a custom LLM runtime implementation in Rust.

---

## ⚠️ Current Development Status

**At the current stage, this project only supports reading and inspecting model metadata, tensor headers, shapes, data types, and raw or F32 weight values.** 

Response generation, tokenization, and model inference are **not yet implemented** and will be available in future releases.

---

## Project Structure

The project consists of the following components:

- **`Cargo.toml`**: Project configuration and dependencies (`serde`, `serde_json`).
- **`src/main.rs`**: Entry point that scans the local `models/` directory, iterates through `.safetensors` files, parses their structures, and prints out metadata, tensor statistics, and sample weights.
- **`src/loader.rs`**: Core parser implementation (`SafeTensorsModel` struct and `TensorInfo`) handling binary header extraction, JSON deserialization, and tensor memory mapping / byte slicing.

---

## Features

- **SafeTensors Parsing**: Reads the 8-length little-endian header length followed by the JSON metadata header.
- **Metadata Extraction**: Automatically extracts the global `__metadata__` map from the safetensors header.
- **Tensor Inspection**: Inspects tensor names, data types (e.g., `F32`), shapes, and byte offsets.
- **Weight Sampling**: Provides safe retrieval methods for raw bytes and `f32` tensor chunks.

---

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/) (Edition 2024 or compatible toolchain)

### Setup & Usage

1. Create a `models` folder in the root directory of the project (next to `src/` and `Cargo.toml`).
2. Place one or more `.safetensors` model files inside the `models/` directory.
3. Run the project using Cargo:

```bash
cargo run
```

---

## Future Roadmap

- [ ] Implement a lightweight Tokenizer (BPE/SentencePiece).
- [ ] Add support for additional tensor data types (e.g., F16, BF16, quantized formats).
- [ ] Implement core transformer layers and self-attention mechanisms.
- [ ] **Enable text response generation and interactive inference.**