use ir::{ops::shape::Squeeze, IntoTensor, Tensor};

#[test]
fn test_squeeze_no_params() {
    let input = Tensor::arange::<f32>(vec![1, 2, 5, 5], 0., 50., 1.);
    let desired = Tensor::arange::<f32>(vec![2, 5, 5], 0., 50., 1.);
    let squeeze = Squeeze { axes: None };
    let output = squeeze.squeeze::<f32>(&input).unwrap().into_tensor();
    assert_eq!(desired, output);
}

#[test]
fn test_squeeze_params() {
    let input = Tensor::arange::<f32>(vec![1, 1, 5, 5], 0., 25., 1.);
    let desired = Tensor::arange::<f32>(vec![5, 5], 0., 25., 1.);
    let squeeze = Squeeze {
        axes: Some(vec![0, 1]),
    };
    let output = squeeze.squeeze::<f32>(&input).unwrap().into_tensor();
    assert_eq!(desired, output);
}
