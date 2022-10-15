use ndarray::array;
use steelix::ops::shape::Gather;
use steelix::prelude::*;
#[test]
fn test_gather() {
    let data = array![[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]].into_arc_tensor();
    let indicies = array![[0_i64, 1], [1, 2]].into_arc_tensor();
    let gather = Gather { axis: 0 };

    let desired_data = array![[[1.0, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]].into_arc_tensor();

    let desired = RealizedOp {
        cost: OpCost::zero_cost(),
        outputs: pvec!(desired_data),
    };

    let output = Op::realize(&gather, pvec!(data, indicies)).expect("Failed to realize squeeze.");
    assert_eq!(desired, output);
}
