pub mod onnx_pb {
    include!(concat!(env!("OUT_DIR"), "/prost/onnx.rs"));
}

mod pb_helpers;

pub use pb_helpers::*;
