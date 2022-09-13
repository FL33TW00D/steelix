use crate::helpers::npy_as_tensor;
use ir::{ops::nn::BatchNormalization, IntoTensor, Tensor};

#[test]
fn test_batchnorm() {
    let input = Tensor::arange::<f32>(vec![2, 144, 3, 3], 0., 2592., 1.);
    let bn = BatchNormalization {
        #[allow(clippy::excessive_precision)]
        epsilon: 0.0010000000474974513,
    };

    let eff_scale = npy_as_tensor(
        "models/eff-lite/batchnorm/eff_scale.npy",
        vec![1, 144, 1, 1],
    );
    let eff_shift = npy_as_tensor("models/eff-lite/batchnorm/eff_B.npy", vec![1, 144, 1, 1]);
    let eff_mean = npy_as_tensor("models/eff-lite/batchnorm/eff_mean.npy", vec![144]);
    let eff_var = npy_as_tensor("models/eff-lite/batchnorm/eff_var.npy", vec![144]);

    let desired = npy_as_tensor(
        "models/eff-lite/batchnorm/eff_bn_output.npy",
        vec![2, 144, 3, 3],
    );
    let output = bn
        .normalize::<f32>(&input, &eff_scale, &eff_shift, &eff_mean, &eff_var)
        .unwrap()
        .into_tensor();
    assert!(desired.all_close(&output, 1e-2));
}
