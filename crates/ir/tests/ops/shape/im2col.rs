use ir::{
    ops::shape::{im2col, Pad},
    IntoTensor, Tensor,
};
use ndarray::Array;

use crate::helpers::npy_as_tensor;

#[test]
fn test_im2col() {
    let input = Tensor::arange::<f32>(vec![1, 3, 224, 224], 0., 150528., 1.);
    let padded = Pad::pad::<f32>(&input, vec![1, 1, 1, 1]).unwrap();
    let (s0, s1): (usize, usize) = (1, 1);
    let kernel_shape: Vec<usize> = vec![1, 1, 3, 3];

    let output: Tensor = im2col::<f32>(&padded, &kernel_shape, (s0, s1), (224, 224)).into();
    println!("OUTPUT: {:?}", output);
    let ground = npy_as_tensor("models/eff-lite/im2col_ground.npy", vec![27, 50176]);
    assert!(output.all_close(&ground, 1e-2))
}
