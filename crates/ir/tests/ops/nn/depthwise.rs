use crate::helpers::npy_as_tensor;
use ir::{ops::nn::Depthwise, IntoTensor, Tensor};
use ndarray::Array;

#[test]
fn depthwise_pytorch() {
    let mut input_vec: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    input_vec.append(&mut vec![6.0_f32; 401402]);

    let input: Tensor = Array::from_shape_vec((1, 32, 112, 112), input_vec)
        .unwrap()
        .into_tensor();
    let weights = npy_as_tensor(
        "models/eff-lite/depthwise/efficientnet-lite4_model_blocks_0_depthwise_conv2d_depthwise_weights_fused_bn.npy",
        vec![32, 1, 3, 3],
    );
    let bias = npy_as_tensor(
        "models/eff-lite/depthwise/efficientnet-lite4_model_blocks_0_depthwise_conv2d_depthwise_bias_fused_bn.npy",
        vec![32],
    );
    let pytorch_output = npy_as_tensor(
        "models/eff-lite/depthwise/eff_depthwise.npy",
        vec![1, 32, 112, 112],
    );

    let depthwise = Depthwise {
        group: 32,
        pads: vec![1, 1, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![1, 1],
    };

    let output = depthwise
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    assert!(output.all_close(&pytorch_output, 1e-2))
}

#[test]
fn depthwise_pytorch_complex() {
    let mut input_vec: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    input_vec.append(&mut vec![6.0_f32; 602106]);

    let input: Tensor = Array::from_shape_vec((1, 192, 56, 56), input_vec)
        .unwrap()
        .into_tensor();
    let weights = npy_as_tensor(
        "models/eff-lite/depthwise/efficientnet-lite4_model_blocks_5_depthwise_conv2d_depthwise_weights_fused_bn.npy",
        vec![192,1,5,5],
    );
    let bias = npy_as_tensor(
        "models/eff-lite/depthwise/efficientnet-lite4_model_blocks_5_depthwise_conv2d_depthwise_bias_fused_bn.npy",
        vec![192],
    );

    let pytorch_output = npy_as_tensor(
        "models/eff-lite/depthwise/eff_depthwise_complex.npy",
        vec![1, 192, 28, 28],
    );

    let depthwise = Depthwise {
        group: 192,
        pads: vec![1, 1, 2, 2],
        kernel_shape: Some(vec![5, 5]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let output = depthwise
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    assert!(output.all_close(&pytorch_output, 1e-2))
}

#[test]
fn simple_depthwise() {
    let input = Tensor::arange::<f32>(vec![1, 4, 3, 3], 0., 36., 1.);
    let weights = Tensor::arange::<f32>(vec![4, 1, 3, 3], 0., 36., 1.);
    let bias: Tensor = Array::from_shape_vec((4,), vec![0.0_f32; 4])
        .unwrap()
        .into_tensor();

    let depthwise = Depthwise {
        group: 4,
        pads: vec![1, 1, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![1, 1],
    };

    let output = depthwise
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    let desired: Vec<f32> = vec![
        58.0, 100.0, 70.0, 132.0, 204.0, 132.0, 70.0, 100.0, 58.0, 670.0, 1018.0, 682.0, 1050.0,
        1581.0, 1050.0, 682.0, 1018.0, 670.0, 1930.0, 2908.0, 1942.0, 2940.0, 4416.0, 2940.0,
        1942.0, 2908.0, 1930.0, 3838.0, 5770.0, 3850.0, 5802.0, 8709.0, 5802.0, 3850.0, 5770.0,
        3838.0,
    ];

    let expected_output: Tensor = Array::from_shape_vec((1, 4, 3, 3), desired)
        .unwrap()
        .into_tensor();

    assert!(output.all_close(&expected_output, 1e-2))
}
