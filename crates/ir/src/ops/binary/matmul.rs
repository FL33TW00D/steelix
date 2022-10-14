use std::borrow::Cow;

use anyhow::{bail, format_err};
use onnx::onnx_pb;

use crate::{
    ops::shape::multi_broadcast, pvec, validate_providers, BoxOp, IntoArcTensor, Op, OpCost,
    OpGroup, PVec, RealizedOp, Shape, Tensor,
};

#[derive(Debug, Clone)]
pub struct Matmul;

impl Op for Matmul {
    fn name(&self) -> Cow<str> {
        "Matmul".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    //If both arguments are 2-D they are multiplied like conventional matrices.
    //If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    //If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
    //If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
    //𝑛𝑚(2𝑝−1)
    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 2, &self.name())?;
        let a_shape = &providers[0].shape;
        let b_shape = &providers[1].shape;
        let c_shape = compute_shapes(a_shape.clone(), b_shape.clone())?;

        let m = a_shape[0];
        let n = b_shape[1];
        let p = a_shape[1];

        let res = Tensor::new(providers[0].dt, c_shape);

        Ok(RealizedOp {
            cost: OpCost {
                flops: m * n * (2 * p - 1),
                parameters: 0,
            },
            outputs: pvec![res.into_arc_tensor()],
        })
    }
}

pub fn build_matmul(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Matmul) as BoxOp)
}

pub fn compute_shapes(mut ashape: Shape, mut bshape: Shape) -> anyhow::Result<Shape> {
    let mut implicit_m = false;
    let mut implicit_n = false;
    if ashape.len() < 2 {
        implicit_m = true;
        ashape.insert(0, 1);
    }
    if bshape.len() < 2 {
        implicit_n = true;
        bshape.insert(0, 1);
    }
    while ashape.len() < bshape.len() {
        ashape.insert(0, 1);
    }
    while bshape.len() < ashape.len() {
        bshape.insert(0, 1);
    }
    let broadcasted_shape = multi_broadcast(&[
        Shape(ashape[..(ashape.len() - 2)].into()),
        Shape(bshape[..(bshape.len() - 2)].into()),
    ])
    .ok_or_else(|| format_err!("Could not broadcast"))?;
    let mut c_shape: Shape = broadcasted_shape;
    let m = ashape[ashape.len() - 2];
    let n = bshape[bshape.len() - 1];
    let mut c_shape_final = c_shape.clone();
    c_shape.push(m);
    c_shape.push(n);
    if !implicit_m {
        c_shape_final.push(m);
    }
    if !implicit_n {
        c_shape_final.push(n);
    }
    Ok(c_shape)
}
