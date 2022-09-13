use crate::helpers::npy_as_tensor;
use ir::{ops::activation::Softmax, Tensor};

#[test]
fn test_softmax() {
    let input = Tensor::arange::<f32>(vec![5, 5], 0., 25., 1.);

    let softmax = Softmax { axis: 1 };

    let out = softmax.softmax::<f32>(&input).unwrap();
    let desired = npy_as_tensor("models/eff-lite/softmax/eff_softmax.npy", vec![5, 5]);
    assert!(desired.all_close(&out, 1e-2))
}
