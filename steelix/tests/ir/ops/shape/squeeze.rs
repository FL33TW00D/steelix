use ir::{ops::shape::Squeeze, pvec, shape, IntoArcTensor, Op, OpCost, RealizedOp, Tensor};

#[test]
fn test_squeeze_no_params() {
    let input = Tensor::zeros::<f32>(shape!(1, 1, 2, 5, 5)).into_arc_tensor();
    let squeeze = Squeeze { axes: None };

    let desired = RealizedOp {
        cost: OpCost::zero_cost(),
        outputs: pvec!(Tensor::zeros::<f32>(shape!(2, 5, 5)).into_arc_tensor()),
    };

    let output = Op::realize(&squeeze, pvec!(input)).expect("Failed to realize squeeze.");
    assert_eq!(desired, output);
}

#[test]
fn test_squeeze_params() {
    let input = Tensor::zeros::<f32>(shape!(1, 1, 5, 5)).into_arc_tensor();
    let squeeze = Squeeze {
        axes: Some(vec![0, 1]),
    };
    let desired = RealizedOp {
        cost: OpCost::zero_cost(),
        outputs: pvec!(Tensor::zeros::<f32>(shape!(5, 5)).into_arc_tensor()),
    };

    let output = Op::realize(&squeeze, pvec!(input)).expect("Failed to realize squeeze.");
    assert_eq!(desired, output);
}
