use std::{fmt::Display, path::PathBuf};

use ir::DType;
use serde::Deserialize;

#[derive(thiserror::Error, Debug)]
pub enum DeviceError {
    #[error("{0}")]
    NumberFormatError(String),
}

///Represents a hardware device for running the ONNX model
#[derive(Deserialize)]
pub struct Device {
    pub name: String,
    pub stats: DeviceStats,
}

#[derive(Deserialize)]
pub struct DeviceStats {
    pub tops: usize,
    pub half: usize,
    pub single: usize,
    pub double: usize,
}

pub struct Iterations(f64);

impl Display for Iterations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} it/s", self.0)
    }
}

impl Device {
    pub fn calculate_its(&self, dt: DType, flops: usize) -> Result<Iterations, DeviceError> {
        let flops_per_sec = match dt {
            DType::I8 => self.stats.tops,
            DType::F16 => self.stats.half,
            DType::F32 => self.stats.single,
            DType::F64 => self.stats.double,
            _ => {
                return Err(DeviceError::NumberFormatError(
                    "Invalid data type provided.".to_string(),
                ))
            }
        };

        Ok(Iterations((flops / flops_per_sec) as f64))
    }
}

pub fn load_devices() -> Result<Vec<Device>, anyhow::Error> {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..2 {
        d.pop();
    }
    d.push("resources");
    d.push("devices");

    let entries = std::fs::read_dir(d)?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    println!("Entries : {:?}", entries);

    let mut devices = vec![];
    for entry in entries {
        let device_str = std::fs::read_to_string(entry)?;

        let device: Device = serde_json::from_str(&device_str)?;
        devices.push(device);
    }

    Ok(devices)
}
