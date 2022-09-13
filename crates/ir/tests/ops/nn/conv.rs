use crate::helpers::npy_as_tensor;
use ir::{ops::nn::Conv, IntoTensor, Tensor};
use ndarray::Array;

#[test]
fn pytorch_conv_comparison() {
    let steelix_input = npy_as_tensor("transpose_out.npy", vec![1, 3, 224, 224]);
    let weights = npy_as_tensor(
        "models/eff-lite/conv/efficientnet-lite4_model_stem_conv2d_Conv2D_weights_fused_bn.npy",
        vec![32, 3, 3, 3],
    );
    let bias = npy_as_tensor(
        "models/eff-lite/conv/efficientnet-lite4_model_stem_conv2d_Conv2D_bias_fused_bn.npy",
        vec![32],
    );
    let pytorch_output = npy_as_tensor("models/eff-lite/conv/eff_conv.npy", vec![1, 32, 112, 112]);

    let conv = Conv {
        group: 1,
        pads: vec![0, 0, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let output = conv
        .convolve::<f32>(&steelix_input, &weights, &bias)
        .unwrap()
        .into_tensor();
    assert!(output.all_close(&pytorch_output, 1e-2))
}

#[test]
fn onnx_backend_simple_conv() {
    let input = Tensor::arange::<f32>(vec![1, 1, 5, 5], 0., 25., 1.);
    let weights: Tensor = Array::from_shape_vec((1, 1, 3, 3), vec![1.0_f32; 9])
        .unwrap()
        .into_tensor();

    let bias: Tensor = Array::from_shape_vec((1,), vec![0.0_f32; 1])
        .unwrap()
        .into_tensor();

    let conv = Conv {
        group: 1,
        pads: vec![1, 1, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![1, 1],
    };

    let output = conv
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    let desired: Vec<f32> = vec![
        12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0, 81.0,
        93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0,
    ];
    let expected_output: Tensor = Array::from_shape_vec((1, 1, 5, 5), desired)
        .unwrap()
        .into_tensor();

    assert_eq!(expected_output, output)
}
#[test]
fn test_strided_conv() {
    let input = Tensor::arange::<f32>(vec![1, 1, 5, 5], 0., 25., 1.);
    let weights: Tensor = Array::from_shape_vec((1, 1, 3, 3), vec![1.0_f32; 9])
        .unwrap()
        .into_tensor();

    let bias: Tensor = Array::from_shape_vec((1,), vec![0.0_f32; 1])
        .unwrap()
        .into_tensor();

    let conv = Conv {
        group: 1,
        pads: vec![0, 0, 0, 0],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let output = conv
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    let desired: Vec<f32> = vec![54., 72., 144., 162.];
    let expected_output: Tensor = Array::from_shape_vec((1, 1, 2, 2), desired)
        .unwrap()
        .into_tensor();

    assert_eq!(expected_output, output)
}

///Tests 2 channels and 2 filters
#[test]
fn test_channels_conv() {
    let input = Tensor::arange::<f32>(vec![1, 2, 5, 5], 0., 50., 1.);
    let weights = Tensor::arange::<f32>(vec![2, 2, 3, 3], 0., 36., 1.);

    let bias: Tensor = Array::from_shape_vec((2,), vec![0.0_f32; 2])
        .unwrap()
        .into_tensor();

    let conv = Conv {
        group: 1,
        pads: vec![1, 1, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![1, 1],
    };

    let output = conv
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    let desired: Vec<f32> = vec![
        1784.0, 2648.0, 2768.0, 2888.0, 1888.0, 2742.0, 4035.0, 4188.0, 4341.0, 2814.0, 3282.0,
        4800.0, 4953.0, 5106.0, 3294.0, 3822.0, 5565.0, 5718.0, 5871.0, 3774.0, 2312.0, 3332.0,
        3416.0, 3500.0, 2224.0, 4016.0, 6104.0, 6440.0, 6776.0, 4552.0, 6630.0, 10029.0, 10506.0,
        10983.0, 7350.0, 8250.0, 12414.0, 12891.0, 13368.0, 8910.0, 9870.0, 14799.0, 15276.0,
        15753.0, 10470.0, 6704.0, 10028.0, 10328.0, 10628.0, 7048.0,
    ];

    let expected_output: Tensor = Array::from_shape_vec((1, 2, 5, 5), desired)
        .unwrap()
        .into_tensor();

    assert_eq!(expected_output, output)
}

///Tests 2 channels and 2 filters, stride 2
#[test]
fn test_strided_channels_conv() {
    let input = Tensor::arange::<f32>(vec![1, 2, 5, 5], 0., 50., 1.);
    let weights = Tensor::arange::<f32>(vec![2, 2, 3, 3], 0., 36., 1.);

    let bias: Tensor = Array::from_shape_vec((2,), vec![0.0_f32; 2])
        .unwrap()
        .into_tensor();

    let conv = Conv {
        group: 1,
        pads: vec![1, 1, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let output = conv
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    let desired: Vec<f32> = vec![
        1784.0, 2768.0, 1888.0, 3282.0, 4953.0, 3294.0, 2312.0, 3416.0, 2224.0, 4016.0, 6440.0,
        4552.0, 8250.0, 12891.0, 8910.0, 6704.0, 10328.0, 7048.0,
    ];

    let expected_output: Tensor = Array::from_shape_vec((1, 2, 3, 3), desired)
        .unwrap()
        .into_tensor();

    assert_eq!(expected_output, output)
}

///Tests 2 channels and 2 filters, stride 2
#[test]
fn test_mid_conv() {
    let input = Tensor::arange::<f32>(vec![2, 2, 5, 5], 0., 100., 1.);
    let weights = Tensor::arange::<f32>(vec![2, 2, 3, 3], 0., 36., 1.);
    let bias: Tensor = Array::from_shape_vec((2,), vec![1.0_f32, 1.0_f32])
        .unwrap()
        .into_tensor();

    let conv = Conv {
        group: 1,
        pads: vec![2, 2, 2, 2],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let output = conv
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    let desired: Vec<f32> = vec![
        426.0, 1274.0, 1412.0, 460.0, 1396.0, 4036.0, 4342.0, 1366.0, 1966.0, 5566.0, 5872.0,
        1816.0, 536.0, 1448.0, 1514.0, 442.0, 876.0, 2732.0, 3086.0, 1054.0, 3286.0, 10030.0,
        10984.0, 3688.0, 4936.0, 14800.0, 15754.0, 5218.0, 1706.0, 5066.0, 5348.0, 1756.0, 1676.0,
        4724.0, 4862.0, 1510.0, 4246.0, 11686.0, 11992.0, 3616.0, 4816.0, 13216.0, 13522.0, 4066.0,
        1186.0, 3098.0, 3164.0, 892.0, 3926.0, 11582.0, 11936.0, 3904.0, 11536.0, 33880.0, 34834.0,
        11338.0, 13186.0, 38650.0, 39604.0, 12868.0, 4156.0, 12116.0, 12398.0, 4006.0,
    ];

    let expected_output: Tensor = Array::from_shape_vec((2, 2, 4, 4), desired)
        .unwrap()
        .into_tensor();

    assert_eq!(expected_output, output)
}

#[test]
fn test_complex_conv() {
    let input = Tensor::arange::<f32>(vec![3, 2, 5, 5], 0., 150., 1.);
    let weights = Tensor::arange::<f32>(vec![1, 2, 3, 3], 0., 18., 1.);

    let bias: Tensor = Array::from_shape_vec((1,), (0u16..1).map(f32::from).collect())
        .unwrap()
        .into_tensor();

    let conv = Conv {
        group: 1,
        pads: vec![2, 2, 2, 2],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let output = conv
        .convolve::<f32>(&input, &weights, &bias)
        .unwrap()
        .into_tensor();

    let desired: Vec<f32> = vec![
        425.0, 1273.0, 1411.0, 459.0, 1395.0, 4035.0, 4341.0, 1365.0, 1965.0, 5565.0, 5871.0,
        1815.0, 535.0, 1447.0, 1513.0, 441.0, 1675.0, 4723.0, 4861.0, 1509.0, 4245.0, 11685.0,
        11991.0, 3615.0, 4815.0, 13215.0, 13521.0, 4065.0, 1185.0, 3097.0, 3163.0, 891.0, 2925.0,
        8173.0, 8311.0, 2559.0, 7095.0, 19335.0, 19641.0, 5865.0, 7665.0, 20865.0, 21171.0, 6315.0,
        1835.0, 4747.0, 4813.0, 1341.0,
    ];

    let expected_output: Tensor = Array::from_shape_vec((3, 1, 4, 4), desired)
        .unwrap()
        .into_tensor();

    assert_eq!(expected_output, output)
}
