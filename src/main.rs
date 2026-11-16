use std::fs;
use std::path::{Path, PathBuf};

mod loader;
mod model;

use model::{Model, RunState};

/// Recursively collects `.safetensors` files up to one subdirectory deep inside `models/`.
fn find_safetensors_files(models_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut results = Vec::new();

    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().is_some_and(|ext| ext == "safetensors") {
            results.push(path);
        } else if path.is_dir() {
            for sub_entry in fs::read_dir(&path)? {
                let sub_entry = sub_entry?;
                let sub_path = sub_entry.path();
                if sub_path.is_file()
                    && sub_path.extension().is_some_and(|ext| ext == "safetensors")
                {
                    results.push(sub_path);
                }
            }
        }
    }

    Ok(results)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models_dir = Path::new("models");

    if !models_dir.exists() || !models_dir.is_dir() {
        eprintln!("Directory {:?} not found.", models_dir);
        return Ok(());
    }

    let safetensors_files = find_safetensors_files(models_dir)?;
    if safetensors_files.is_empty() {
        eprintln!("No .safetensors files found in {:?}.", models_dir);
        return Ok(());
    }

    // Load the first discovered model
    let weights_path = &safetensors_files[0];
    let config_path = weights_path
        .parent()
        .map(|dir| dir.join("config.json"))
        .filter(|p| p.exists());

    println!("Loading model from: {:?}", weights_path);
    if let Some(cfg) = &config_path {
        println!("Using config file: {:?}", cfg);
    }

    let model = Model::load(weights_path, config_path.as_deref())?;

    // Allocate intermediate computation buffers (cap context length at 2048 by default)
    let target_seq_len = 2048;
    let mut state = RunState::new(&model.config, target_seq_len);

    let weights_mb = model.weights.memory_bytes() as f64 / (1024.0 * 1024.0);
    let state_mb = state.memory_bytes() as f64 / (1024.0 * 1024.0);

    println!("\n=== Model Loaded Successfully ===");
    println!("Vocab size:         {}", model.config.vocab_size);
    println!("Hidden size:        {}", model.config.hidden_size);
    println!("Intermediate size:  {}", model.config.intermediate_size);
    println!("Layers:             {}", model.config.num_hidden_layers);
    println!(
        "Attention heads:    {} (KV heads: {}, head_dim: {})",
        model.config.num_attention_heads, model.config.num_key_value_heads, model.config.head_dim
    );
    println!("Tied embeddings:    {}", model.config.tie_word_embeddings);
    println!("Weights RAM (f32):  {:.2} MB", weights_mb);
    println!(
        "RunState RAM:       {:.2} MB (max_seq_len = {})",
        state_mb, state.max_seq_len
    );

    // Ensure scratchpad is ready for the first forward pass
    state.reset();

    Ok(())
}