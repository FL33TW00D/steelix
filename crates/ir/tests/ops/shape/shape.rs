use ir::{ops::shape::Shape, pvec, shape, IntoArcTensor, Op, OpCost, RealizedOp, Tensor};

#[test]
fn test_shape() {
    let shape_op = Shape { start: 0, end: -1 };

    let input_tensor = Tensor::zeros::<f32>(shape!(2, 3, 4)).into_arc_tensor();

    let values = vec![2_i64, 3, 4];
    let desired_tensor = Tensor::from_vec(shape![3], values).into_arc_tensor();

    let desired = RealizedOp {
        cost: OpCost::zero_cost(),
        outputs: pvec!(desired_tensor),
    };

    let output = Op::realize(&shape_op, pvec!(input_tensor)).expect("Failed to realize squeeze.");
    assert_eq!(desired, output);
}
