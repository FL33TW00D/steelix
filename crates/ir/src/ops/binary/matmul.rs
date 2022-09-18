use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, RealizedOp, Tensor};

#[derive(Debug, Clone)]
pub struct Matmul;

impl Op for Matmul {
    fn name(&self) -> Cow<str> {
        "Matmul".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    fn realize(&self, providers: crate::PVec) -> anyhow::Result<crate::RealizedOp> {
        let p0_shape = &providers[0].shape;
        let p1_shape = &providers[1].shape;

        let output_shape = vec![p0_shape[0], p1_shape[1]];

        let res = Tensor::new(providers[0].dt, output_shape.into());

        Ok(RealizedOp {
            cost: OpCost {
                flops: providers[0].numel(),
                parameters: 0,
            },
            outputs: smallvec![res.into_arc_tensor(); 4],
        })
    }
}

pub fn build_matmul(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Matmul) as BoxOp)
}
