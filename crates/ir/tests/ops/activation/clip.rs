use ir::{ops::activation::Clip, IntoTensor, Tensor};
use ndarray::Array;

#[test]
fn clip() {
    let min: Tensor = Array::from_shape_vec((1,), vec![0.0_f32])
        .unwrap()
        .into_tensor();

    let max: Tensor = Array::from_shape_vec((1,), vec![6.0_f32])
        .unwrap()
        .into_tensor();

    let input = Tensor::arange::<f32>(vec![2, 2, 8, 8], 0., 256., 1.);
    println!("INPUT: {:?}", input);

    let clip = Clip {
        min: None,
        max: None,
    };

    let clipped = clip.clip::<f32>(&input, &min, &max).unwrap().into_tensor();

    let mut desired_vec: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    desired_vec.append(&mut vec![6.0_f32; 250]);

    let desired: Tensor = Array::from_shape_vec((2, 2, 8, 8), desired_vec)
        .unwrap()
        .into_tensor();

    assert_eq!(desired, clipped)
}
