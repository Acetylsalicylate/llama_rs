use crate::loader::SafeTensorsModel;
use serde::Deserialize;
use std::fs;
use std::io;
use std::path::Path;

/// Raw HuggingFace config.json schema for Llama / Qwen2 style architectures.
#[derive(Debug, Deserialize)]
struct RawConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    tie_word_embeddings: Option<bool>,
}

/// Normalized Transformer architecture configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Total query projection dimension (`num_attention_heads * head_dim`).
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Total key/value projection dimension (`num_key_value_heads * head_dim`).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Loads model configuration from a HuggingFace `config.json` file.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let raw: RawConfig = serde_json::from_str(&content)?;

        let num_key_value_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);

        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings: raw.max_position_embeddings.unwrap_or(2048),
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: raw.rope_theta.unwrap_or(10_000.0),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        })
    }

    /// Fallback configuration inference from SafeTensors tensor shapes if `config.json` is missing.
    pub fn infer_from_safetensors(st: &SafeTensorsModel) -> Result<Self, Box<dyn std::error::Error>> {
        let embed = st.tensors.get("model.embed_tokens.weight").ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "Missing 'model.embed_tokens.weight' tensor",
            )
        })?;
        let vocab_size = embed.shape[0];
        let hidden_size = embed.shape[1];

        let gate = st
            .tensors
            .get("model.layers.0.mlp.gate_proj.weight")
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "Missing 'model.layers.0.mlp.gate_proj.weight' tensor",
                )
            })?;
        let intermediate_size = gate.shape[0];

        let mut num_hidden_layers = 0usize;
        while st
            .contains_tensor(&format!("model.layers.{num_hidden_layers}.input_layernorm.weight"))
        {
            num_hidden_layers += 1;
        }
        if num_hidden_layers == 0 {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                "Could not detect any transformer layers",
            )));
        }

        let q_proj = st
            .tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "Missing 'model.layers.0.self_attn.q_proj.weight' tensor",
                )
            })?;
        let k_proj = st
            .tensors
            .get("model.layers.0.self_attn.k_proj.weight")
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "Missing 'model.layers.0.self_attn.k_proj.weight' tensor",
                )
            })?;

        let q_dim = q_proj.shape[0];
        let kv_dim = k_proj.shape[0];

        // Standard head_dim heuristic when config.json is not provided
        let head_dim = if kv_dim % 64 == 0 && q_dim % 64 == 0 {
            64
        } else if kv_dim % 128 == 0 && q_dim % 128 == 0 {
            128
        } else {
            kv_dim
        };

        let num_attention_heads = q_dim / head_dim;
        let num_key_value_heads = kv_dim / head_dim;
        let tie_word_embeddings = !st.contains_tensor("lm_head.weight");

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings,
        })
    }
}

/// Weights for a single Transformer decoder block.
pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,          // [hidden_size]
    pub q_proj: Vec<f32>,                   // [q_dim * hidden_size]
    pub q_bias: Option<Vec<f32>>,           // [q_dim] (present in Qwen2/Qwen2.5)
    pub k_proj: Vec<f32>,                   // [kv_dim * hidden_size]
    pub k_bias: Option<Vec<f32>>,           // [kv_dim]
    pub v_proj: Vec<f32>,                   // [kv_dim * hidden_size]
    pub v_bias: Option<Vec<f32>>,           // [kv_dim]
    pub o_proj: Vec<f32>,                   // [hidden_size * q_dim]
    pub post_attention_layernorm: Vec<f32>, // [hidden_size]
    pub gate_proj: Vec<f32>,                // [intermediate_size * hidden_size]
    pub up_proj: Vec<f32>,                  // [intermediate_size * hidden_size]
    pub down_proj: Vec<f32>,                // [hidden_size * intermediate_size]
}

impl LayerWeights {
    /// Returns total heap memory used by this layer's weights in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut elems = self.input_layernorm.len()
            + self.q_proj.len()
            + self.k_proj.len()
            + self.v_proj.len()
            + self.o_proj.len()
            + self.post_attention_layernorm.len()
            + self.gate_proj.len()
            + self.up_proj.len()
            + self.down_proj.len();

        if let Some(b) = &self.q_bias {
            elems += b.len();
        }
        if let Some(b) = &self.k_bias {
            elems += b.len();
        }
        if let Some(b) = &self.v_bias {
            elems += b.len();
        }

        elems * size_of::<f32>()
    }
}

/// All Transformer weights loaded in memory as f32.
pub struct TransformerWeights {
    pub embed_tokens: Vec<f32>,    // [vocab_size * hidden_size]
    pub layers: Vec<LayerWeights>, // [num_hidden_layers]
    pub norm: Vec<f32>,            // [hidden_size]
    pub lm_head: Option<Vec<f32>>, // [vocab_size * hidden_size], None when tied to embed_tokens
}

impl TransformerWeights {
    /// Returns the output projection weight slice, falling back to `embed_tokens` when tied.
    #[inline]
    pub fn lm_head_weights(&self) -> &[f32] {
        self.lm_head.as_deref().unwrap_or(&self.embed_tokens)
    }

    /// Returns total heap memory used by all model weights in bytes.
    pub fn memory_bytes(&self) -> usize {
        let layers_bytes: usize = self.layers.iter().map(|l| l.memory_bytes()).sum();
        let top_elems = self.embed_tokens.len()
            + self.norm.len()
            + self.lm_head.as_ref().map_or(0, |w| w.len());

        layers_bytes + top_elems * size_of::<f32>()
    }
}

/// Complete loaded model containing architecture configuration and weights.
pub struct Model {
    pub config: ModelConfig,
    pub weights: TransformerWeights,
}

impl Model {
    /// Loads a model from a `.safetensors` file and an optional `config.json` file.
    pub fn load<P: AsRef<Path>>( safetensors_path: P, config_path: Option<&Path>, ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = SafeTensorsModel::parse(&safetensors_path)?;

        let config = match config_path {
            Some(path) if path.exists() => ModelConfig::from_json_file(path)?,
            _ => {
                eprintln!(
                    "Warning: config.json not found, inferring architecture from tensor shapes."
                );
                ModelConfig::infer_from_safetensors(&st)?
            }
        };

        let mut file = st.open_file()?;

        let embed_tokens = load_checked(
            &st,
            &mut file,
            "model.embed_tokens.weight",
            config.vocab_size * config.hidden_size,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let input_layernorm = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden_size,
            )?;

            let q_proj = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.self_attn.q_proj.weight"),
                q_dim * config.hidden_size,
            )?;
            let q_bias = load_optional_checked(
                &st,
                &mut file,
                &format!("{prefix}.self_attn.q_proj.bias"),
                q_dim,
            )?;

            let k_proj = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.self_attn.k_proj.weight"),
                kv_dim * config.hidden_size,
            )?;
            let k_bias = load_optional_checked(
                &st,
                &mut file,
                &format!("{prefix}.self_attn.k_proj.bias"),
                kv_dim,
            )?;

            let v_proj = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.self_attn.v_proj.weight"),
                kv_dim * config.hidden_size,
            )?;
            let v_bias = load_optional_checked(
                &st,
                &mut file,
                &format!("{prefix}.self_attn.v_proj.bias"),
                kv_dim,
            )?;

            let o_proj = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.self_attn.o_proj.weight"),
                config.hidden_size * q_dim,
            )?;

            let post_attention_layernorm = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?;

            let gate_proj = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.mlp.gate_proj.weight"),
                config.intermediate_size * config.hidden_size,
            )?;
            let up_proj = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.mlp.up_proj.weight"),
                config.intermediate_size * config.hidden_size,
            )?;
            let down_proj = load_checked(
                &st,
                &mut file,
                &format!("{prefix}.mlp.down_proj.weight"),
                config.hidden_size * config.intermediate_size,
            )?;

            layers.push(LayerWeights {
                input_layernorm,
                q_proj,
                q_bias,
                k_proj,
                k_bias,
                v_proj,
                v_bias,
                o_proj,
                post_attention_layernorm,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let norm = load_checked(&st, &mut file, "model.norm.weight", config.hidden_size)?;

        let lm_head = if !config.tie_word_embeddings && st.contains_tensor("lm_head.weight") {
            Some(load_checked(
                &st,
                &mut file,
                "lm_head.weight",
                config.vocab_size * config.hidden_size,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            weights: TransformerWeights {
                embed_tokens,
                layers,
                norm,
                lm_head,
            },
        })
    }
}

/// Pre-allocated scratchpad memory for intermediate Transformer forward-pass computations.
pub struct RunState {
    pub max_seq_len: usize,

    // Residual stream and normalized branch activations
    pub x: Vec<f32>,   // [hidden_size]
    pub xb: Vec<f32>,  // [hidden_size]
    pub xb2: Vec<f32>, // [hidden_size]

    // SwiGLU MLP intermediate buffers
    pub hb: Vec<f32>,  // [intermediate_size]
    pub hb2: Vec<f32>, // [intermediate_size]

    // Attention Q/K/V current-step projections and attention probabilities
    pub q: Vec<f32>,   // [q_dim]
    pub k: Vec<f32>,   // [kv_dim]
    pub v: Vec<f32>,   // [kv_dim]
    pub att: Vec<f32>, // [num_attention_heads * max_seq_len]

    // Output logits over vocabulary
    pub logits: Vec<f32>, // [vocab_size]

    // KV cache across all layers: [num_hidden_layers * max_seq_len * kv_dim]
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,

    // Precomputed RoPE cosine and sine tables: [max_seq_len * (head_dim / 2)]
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
}

impl RunState {
    /// Allocates all intermediate buffers and precomputes RoPE tables for `max_seq_len`.
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> Self {
        let effective_seq_len = max_seq_len.min(config.max_position_embeddings);
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let kv_cache_size = config.num_hidden_layers * effective_seq_len * kv_dim;
        let half_head_dim = config.head_dim / 2;

        // Precompute RoPE frequencies in f64 for accuracy at long contexts
        let mut rope_cos = Vec::with_capacity(effective_seq_len * half_head_dim);
        let mut rope_sin = Vec::with_capacity(effective_seq_len * half_head_dim);
        let theta = config.rope_theta as f64;
        let head_dim_f64 = config.head_dim as f64;

        for pos in 0..effective_seq_len {
            let pos_f64 = pos as f64;
            for i in 0..half_head_dim {
                let freq = 1.0 / theta.powf((2 * i) as f64 / head_dim_f64);
                let angle = pos_f64 * freq;
                rope_cos.push(angle.cos() as f32);
                rope_sin.push(angle.sin() as f32);
            }
        }

        Self {
            max_seq_len: effective_seq_len,
            x: vec![0.0; config.hidden_size],
            xb: vec![0.0; config.hidden_size],
            xb2: vec![0.0; config.hidden_size],
            hb: vec![0.0; config.intermediate_size],
            hb2: vec![0.0; config.intermediate_size],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; config.num_attention_heads * effective_seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; kv_cache_size],
            value_cache: vec![0.0; kv_cache_size],
            rope_cos,
            rope_sin,
        }
    }

    /// Clears activation buffers and KV cache before starting a new sequence.
    pub fn reset(&mut self) {
        self.x.fill(0.0);
        self.xb.fill(0.0);
        self.xb2.fill(0.0);
        self.hb.fill(0.0);
        self.hb2.fill(0.0);
        self.q.fill(0.0);
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.att.fill(0.0);
        self.logits.fill(0.0);
        self.key_cache.fill(0.0);
        self.value_cache.fill(0.0);
    }

    /// Returns total heap memory allocated for intermediate computations in bytes.
    pub fn memory_bytes(&self) -> usize {
        let total_elems = self.x.len()
            + self.xb.len()
            + self.xb2.len()
            + self.hb.len()
            + self.hb2.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.att.len()
            + self.logits.len()
            + self.key_cache.len()
            + self.value_cache.len()
            + self.rope_cos.len()
            + self.rope_sin.len();

        total_elems * size_of::<f32>()
    }
}

/// Loads a required tensor and validates its total element count.
fn load_checked(
    st: &SafeTensorsModel,
    file: &mut fs::File,
    name: &str,
    expected_elems: usize,
) -> io::Result<Vec<f32>> {
    let data = st.read_tensor_f32(file, name)?;
    if data.len() != expected_elems {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Tensor '{name}' element count mismatch: expected {expected_elems}, got {}",
                data.len()
            ),
        ));
    }
    Ok(data)
}

/// Loads an optional tensor if present and validates its element count.
fn load_optional_checked(
    st: &SafeTensorsModel,
    file: &mut fs::File,
    name: &str,
    expected_elems: usize,
) -> io::Result<Option<Vec<f32>>> {
    if !st.contains_tensor(name) {
        return Ok(None);
    }
    let data = load_checked(st, file, name, expected_elems)?;
    Ok(Some(data))
}