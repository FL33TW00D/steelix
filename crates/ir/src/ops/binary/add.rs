use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

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

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp {
            cost: OpCost {
                mac: providers[0].numel(),
                parameters: 0,
            },
            outputs: smallvec![providers[0].clone(); 4],
        })
    }
}

pub fn build_add(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Add) as BoxOp)
}
