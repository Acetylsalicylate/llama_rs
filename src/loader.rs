use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Tensor metadata parsed from the SafeTensors JSON header.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2], // [start, end] offsets relative to the data section
}

impl TensorInfo {
    /// Returns the total number of elements described by `shape`.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Parsed SafeTensors file header and location metadata.
#[derive(Debug)]
pub struct SafeTensorsModel {
    pub path: PathBuf,
    pub metadata: HashMap<String, String>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_start_offset: u64, // Byte offset where tensor data begins
}

/// Converts a single IEEE 754 bfloat16 value (stored as u16) to f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts a single IEEE 754 float16 value (stored as u16) to f32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal f16 -> normalized f32
            let mut e = 127 - 14;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            (sign << 31) | (e << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        // Infinity or NaN
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        // Normal number
        (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13)
    };

    f32::from_bits(f32_bits)
}

impl SafeTensorsModel {
    /// Parses only the SafeTensors header without loading tensor payloads into RAM.
    pub fn parse<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path_buf = path.as_ref().to_path_buf();
        let mut file = File::open(&path_buf)?;

        // Read 8-byte little-endian header length
        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf)?;
        let header_len = u64::from_le_bytes(header_len_buf);
        let header_len_usize = usize::try_from(header_len)?;

        // Read JSON header bytes
        let mut header_bytes = vec![0u8; header_len_usize];
        file.read_exact(&mut header_bytes)?;

        // Deserialize JSON header
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

    /// Checks whether a tensor exists in the SafeTensors header.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Opens the underlying file for sequential/random tensor reads.
    pub fn open_file(&self) -> io::Result<File> {
        File::open(&self.path)
    }

    /// Reads raw bytes of a tensor using an already opened file handle.
    pub fn read_tensor_raw_bytes(&self, file: &mut File, name: &str) -> io::Result<Vec<u8>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Tensor '{name}' not found"))
        })?;

        let length = info.data_offsets[1]
            .checked_sub(info.data_offsets[0])
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid offsets for tensor '{name}'"),
                )
            })?;

        let start_pos = self.data_start_offset + info.data_offsets[0] as u64;
        file.seek(SeekFrom::Start(start_pos))?;

        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    /// Reads raw bytes of a tensor by opening the file for this single read.
    pub fn get_tensor_raw_bytes(&self, name: &str) -> io::Result<Vec<u8>> {
        let mut file = self.open_file()?;
        self.read_tensor_raw_bytes(&mut file, name)
    }

    /// Reads a tensor (BF16, F16, or F32) and converts it into a Vec<f32> using an open file handle.
    pub fn read_tensor_f32(&self, file: &mut File, name: &str) -> io::Result<Vec<f32>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Tensor '{name}' not found"))
        })?;

        let raw = self.read_tensor_raw_bytes(file, name)?;
        let expected_elems = info.num_elements();

        let values: Vec<f32> = match info.dtype.as_str() {
            "F32" => {
                if raw.len() != expected_elems * 4 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Byte size mismatch for F32 tensor '{name}'"),
                    ));
                }
                raw.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            "BF16" => {
                if raw.len() != expected_elems * 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Byte size mismatch for BF16 tensor '{name}'"),
                    ));
                }
                raw.chunks_exact(2)
                    .map(|chunk| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect()
            }
            "F16" => {
                if raw.len() != expected_elems * 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Byte size mismatch for F16 tensor '{name}'"),
                    ));
                }
                raw.chunks_exact(2)
                    .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect()
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unsupported dtype '{other}' for tensor '{name}'"),
                ));
            }
        };

        Ok(values)
    }

    /// Reads and converts a tensor (BF16, F16, or F32) into Vec<f32>.
    pub fn get_tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let mut file = self.open_file()?;
        self.read_tensor_f32(&mut file, name)
    }
}