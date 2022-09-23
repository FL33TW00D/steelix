use ir::{ops::shape::Reshape, pvec, shape, IntoArcTensor, Op, OpCost, RealizedOp, Tensor};
use ndarray::array;

#[test]
fn test_reshape() {
    let data = Tensor::zeros::<f32>(shape!(1, 1280, 1, 1)).into_arc_tensor();
    let new_shape = array![[1280_i64, 1]].into_arc_tensor();
    let reshape = Reshape { allow_zero: 0 };

    let desired_tensor = Tensor::zeros::<f32>(shape!(1280, 1)).into_arc_tensor();

    let desired = RealizedOp {
        cost: OpCost::zero_cost(),
        outputs: pvec!(desired_tensor),
    };

    let output = Op::realize(&reshape, pvec!(data, new_shape)).expect("Failed to realize squeeze.");
    assert_eq!(desired, output);
}
