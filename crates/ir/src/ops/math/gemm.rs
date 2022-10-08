use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp, Tensor};

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
    fn realize(&self, providers: PVec) -> anyhow::Result<crate::RealizedOp> {
        println!("GEMM PROVIDERS: {:?}", providers);
        let p0_shape = &providers[0].shape;
        let p1_shape = &providers[1].shape;
        println!("p0_shape: {:?}", p0_shape);
        println!("p1_shape: {:?}", p1_shape);

        let output_shape = vec![p0_shape[0], p1_shape[1]];

        let m = p0_shape[0];
        let n = p1_shape[1];
        let p = p0_shape[1];

        let res = Tensor::new(providers[0].dt, output_shape.into());

        Ok(RealizedOp {
            cost: OpCost {
                flops: m * n * (2 * p - 1),
                parameters: 0,
            },
            outputs: smallvec![res.into_arc_tensor()],
        })
    }
}

pub fn build_gemm(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Gemm) as BoxOp)
}
