use std::{fmt::Display, path::PathBuf};

use ir::DType;

#[derive(thiserror::Error, Debug)]
pub enum DeviceError {
    #[error("{0}")]
    NumberFormatError(&str),
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
                    "Invalid data type provided.",
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

    let device_paths = std::fs::read_dir(d)?;

    Ok(vec![])
}
