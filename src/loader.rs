use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Description from json header
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2], // [beginning, end] offset from section data
}

/// Parsed struct of SafeTensors
#[derive(Debug)]
pub struct SafeTensorsModel {
    pub path: PathBuf,
    pub metadata: HashMap<String, String>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_start_offset: u64, // first weight byte
}

impl SafeTensorsModel {
    /// Reading onle header
    pub fn parse<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path_buf = path.as_ref().to_path_buf();
        let mut file = File::open(&path_buf)?;

        // read first 8 byte
        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf)?;
        let header_len = u64::from_le_bytes(header_len_buf);

        // reading json header
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)?;

        // deserializing json
        let raw_map: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)?;

        let mut metadata = HashMap::new();
        let mut tensors = HashMap::new();

        for (key, val) in raw_map {
            if key == "__metadata__" {
                if let Ok(meta_map) = serde_json::from_value::<HashMap<String, String>>(val) {
                    metadata = meta_map;
                }
            } else {
                let tensor_info: TensorInfo = serde_json::from_value(val)?;
                tensors.insert(key, tensor_info);
            }
        }

        let data_start_offset = 8 + header_len;

        Ok(Self {
            path: path_buf,
            metadata,
            tensors,
            data_start_offset,
        })
    }

    /// extracting raw bytes 
    pub fn get_tensor_raw_bytes(&self, name: &str) -> io::Result<Vec<u8>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Tensor '{name}' not found"))
        })?;

        let mut file = File::open(&self.path)?;
        let start_pos = self.data_start_offset + info.data_offsets[0] as u64;
        let length = info.data_offsets[1] - info.data_offsets[0];

        file.seek(SeekFrom::Start(start_pos))?;
        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    /// example of parsing f32 tensor
    pub fn get_tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Tensor '{name}' not found"))
        })?;

        if info.dtype != "F32" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("F32 expected, found: {}", info.dtype),
            ));
        }

        let raw = self.get_tensor_raw_bytes(name)?;
        let f32_vec = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(f32_vec)
    }
}