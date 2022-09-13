use std::{path::PathBuf};

use ir::{IntoTensor, Tensor};
use ndarray::Array;
use npyz::{NpyFile};

//TODO: make generic over dtype
pub fn npy_as_tensor(fname: &str, shape: Vec<usize>) -> Tensor {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..2 {
        d.pop();
    }
    d.push("resources");
    d.push(fname);
    let bytes = std::fs::read(d).unwrap();
    let npy = NpyFile::new(&bytes[..]).unwrap();
    let mut data = Vec::new();
    for num in npy.data::<f32>().unwrap() {
        data.push(num.unwrap())
    }

    Array::from_shape_vec(shape, data).unwrap().into_tensor()
}
