use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpCost, OpGroup, QuadVec, RealizedOp};

#[derive(Debug, Clone)]
pub struct Add;

impl Op for Add {
    fn name(&self) -> Cow<str> {
        "Add".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Tensor
    }

    fn cost(&self, inputs: crate::QuadVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp {
            cost: OpCost {
                mac: 0,
                parameters: 0,
                flops: 0,
            },
            outputs: QuadVec::new(),
        })
    }
}

pub fn build_add(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Add) as BoxOp)
}
