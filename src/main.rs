use std::fs::{self};
use std::path::{Path};

mod loader;
use loader::SafeTensorsModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //  models must be near src
    let models_dir = Path::new("models");

    if !models_dir.exists() || !models_dir.is_dir() {
        eprintln!("Папка {:?} не найдена рядом с src!", models_dir);
        return Ok(());
    }

    println!("Сканирование папки {:?}...\n", models_dir);

    // for every file in models
    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();

        // .safetensors or .bin
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    println!("==================================================");
                    println!("File: {:?}", path.file_name().unwrap());

                    match SafeTensorsModel::parse(&path) {
                        Ok(model) => {
                            println!("Metadata: {:?}", model.metadata);
                            println!("Total tensors: {}", model.tensors.len());

                            // Info of firs 5 tensors
                            for (name, info) in model.tensors.iter().take(5) {
                                println!(
                                    " • Tensor: {:<30} | Type: {:<5} | Form: {:?}",
                                    name, info.dtype, info.shape
                                );
                            }

                            // reading first f32 tensor (if has)
                            if let Some((name, _)) = model.tensors.iter().find(|(_, info)| info.dtype == "F32") {
                                if let Ok(values) = model.get_tensor_f32(name) {
                                    println!("\nExample of reading tensor weights '{}':", name);
                                    println!("First 5 value: {:?}", &values[..values.len().min(5)]);
                                }
                            }
                        }
                        Err(e) => eprintln!("Parsing error {:?}: {}", path, e),
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}