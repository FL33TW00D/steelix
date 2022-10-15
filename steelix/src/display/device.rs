use std::{fmt::Display, path::PathBuf};

use crate::ir::DType;
use lazy_static::lazy_static;
use serde::Deserialize;

lazy_static! {
    static ref A100: Device = Device {
        name: "A100".to_string(),
        stats: DeviceStats {
            tops: 6240000000000000,
            half: 3120000000000000,
            single: 1950000000000000,
            double: 970000000000000
        }
    };
    static ref RPI: Device = Device {
        name: "Raspberry Pi 4B".to_string(),
        stats: DeviceStats {
            tops: 0,
            half: 0,
            single: 135000000000,
            double: 0,
        }
    };
}

#[derive(thiserror::Error, Debug)]
pub enum DeviceError {
    #[error("{0}")]
    NumberFormatError(String),
}

///Represents a hardware device for running the ONNX model
#[derive(Deserialize, Clone)]
pub struct Device {
    pub name: String,
    pub stats: DeviceStats,
}

#[derive(Deserialize, Clone)]
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

        Ok(Iterations((flops_per_sec / flops) as f64))
    }
}

pub fn load_devices() -> Result<Vec<Device>, anyhow::Error> {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..1 {
        d.pop();
    }
    d.push("resources");
    d.push("devices");

    if !d.exists() {
        return Ok(vec![A100.clone(), RPI.clone()]);
    }

    let entries = std::fs::read_dir(d)?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, std::io::Error>>()?;

    let devices: Result<Vec<Device>, anyhow::Error> =
        entries.iter().try_fold(Vec::new(), |mut acc, entry| {
            let device = serde_json::from_str(&std::fs::read_to_string(entry)?)?;
            acc.push(device);
            Ok(acc)
        });

    devices
}
