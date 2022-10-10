use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{
    ops::shape::multi_broadcast, pvec, BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp,
    Tensor,
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

    //𝑛𝑚(2𝑝−1)
    fn realize(&self, providers: PVec) -> anyhow::Result<crate::RealizedOp> {
        let broadcasted_shape = multi_broadcast(
            &providers
                .iter()
                .map(|p| p.shape.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let p0_shape = &broadcasted_shape;
        let p1_shape = &broadcasted_shape;

        let m = p0_shape[0];
        let n = p1_shape[1];
        let p = p0_shape[1];

        let res = Tensor::new(providers[0].dt, broadcasted_shape);

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
