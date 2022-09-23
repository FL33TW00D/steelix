use ir::{ops::shape::Squeeze, shape, Tensor};

#[test]
fn test_squeeze_no_params() {
    let input = Tensor::zeros::<f32>(shape!(1, 1, 2, 5, 5));
    let desired_shape = shape!(2, 5, 5);
    let squeeze = Squeeze { axes: None };
    let output = squeeze.squeeze(&input);
    assert_eq!(desired_shape, output);
}

#[test]
fn test_squeeze_params() {
    let input = Tensor::zeros::<f32>(shape![1, 1, 5, 5]);
    let desired_shape = shape!(5, 5);
    let squeeze = Squeeze {
        axes: Some(vec![0, 1]),
    };
    let output = squeeze.squeeze(&input);
    assert_eq!(desired_shape, output);
}
