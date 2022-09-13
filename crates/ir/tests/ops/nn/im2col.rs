use crate::helpers::npy_as_tensor;
use ir::{
    ops::nn::{Conv, Im2Col},
    IntoTensor,
};

#[test]
fn compare_naive() {
    let steelix_input = npy_as_tensor("transpose_out.npy", vec![1, 3, 224, 224]);
    let weights = npy_as_tensor(
        "models/eff-lite/conv/efficientnet-lite4_model_stem_conv2d_Conv2D_weights_fused_bn.npy",
        vec![32, 3, 3, 3],
    );
    let bias = npy_as_tensor(
        "models/eff-lite/conv/efficientnet-lite4_model_stem_conv2d_Conv2D_bias_fused_bn.npy",
        vec![32],
    );

    let conv = Conv {
        group: 1,
        pads: vec![0, 0, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let im2col = Im2Col {
        group: 1,
        pads: vec![0, 0, 1, 1],
        kernel_shape: Some(vec![3, 3]),
        dilations: vec![1, 1],
        strides: vec![2, 2],
    };

    let conv_output = conv
        .convolve::<f32>(&steelix_input, &weights, &bias)
        .unwrap()
        .into_tensor();
    println!("CONV SHAPE: {:?}", conv_output);
    let im2col_output = im2col
        .convolve::<f32>(&steelix_input, &weights, Some(&bias))
        .unwrap()
        .into_tensor();
    println!("IM2COL SHAPE: {:?}", im2col_output);

    assert!(im2col_output.all_close(&conv_output, 1e-2))
}
