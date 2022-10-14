use ir::{ops::shape::Unsqueeze, pvec, shape, IntoArcTensor, Op, OpCost, RealizedOp, Tensor};
use ndarray::array;

#[test]
fn test_unsqueeze_no_params() {
    let input = Tensor::zeros::<f32>(shape!(3, 4, 5)).into_arc_tensor();
    let to_insert = array![0_i64, 4].into_arc_tensor();
    let unsqueeze = Unsqueeze { axes: None };

    let desired = RealizedOp {
        cost: OpCost::zero_cost(),
        outputs: pvec!(Tensor::zeros::<f32>(shape!(1, 3, 4, 5, 1)).into_arc_tensor()),
    };

    let output =
        Op::realize(&unsqueeze, pvec!(input, to_insert)).expect("Failed to realize squeeze.");
    assert_eq!(desired, output);
}

#[test]
fn test_unsqueeze_params() {
    let input = Tensor::zeros::<f32>(shape!(3, 4, 5)).into_arc_tensor();
    let unsqueeze = Unsqueeze {
        axes: Some(vec![0, 4]),
    };

    let desired = RealizedOp {
        cost: OpCost::zero_cost(),
        outputs: pvec!(Tensor::zeros::<f32>(shape!(1, 3, 4, 5, 1)).into_arc_tensor()),
    };

    let output = Op::realize(&unsqueeze, pvec!(input)).expect("Failed to realize squeeze.");
    assert_eq!(desired, output);
}
