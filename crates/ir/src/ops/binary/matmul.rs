use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpGroup, RealizedOp};

#[derive(Debug, Clone)]
pub struct Matmul;

impl Op for Matmul {
    fn name(&self) -> Cow<str> {
        "Matmul".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    fn cost(&self, providers: crate::QuadVec) -> anyhow::Result<crate::RealizedOp> {
        Ok(RealizedOp::default())
    }
}

pub fn build_matmul(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Matmul) as BoxOp)
}
