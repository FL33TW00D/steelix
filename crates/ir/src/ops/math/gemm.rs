use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, RealizedOp, Tensor};

#[derive(Debug, Clone)]
pub struct Gemm;

impl Op for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    //𝑛𝑚(2𝑝−1)
    fn realize(&self, providers: crate::PVec) -> anyhow::Result<crate::RealizedOp> {
        let p0_shape = &providers[0].shape;
        let p1_shape = &providers[1].shape;

        let output_shape = vec![p0_shape[0], p1_shape[1]];

        let m = p0_shape[0];
        let n = p1_shape[1];
        let p = p0_shape[1];

        let flops = m * n * (2 * p - 1);

        let res = Tensor::new(providers[0].dt, output_shape.into(), None);

        Ok(RealizedOp {
            cost: OpCost {
                flops,
                parameters: 0,
            },
            outputs: smallvec![res.into_arc_tensor()],
        })
    }
}

pub fn build_gemm(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Gemm) as BoxOp)
}
