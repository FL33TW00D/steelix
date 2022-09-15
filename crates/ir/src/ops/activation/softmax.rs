use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpCost, OpGroup, QuadVec, RealizedOp};

#[derive(Debug, Clone)]
pub struct Softmax {
    pub axis: i64,
}

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Activation
    }

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp {
            cost: OpCost {
                mac: 0,
                parameters: 0,
            },
            outputs: QuadVec::new(),
        })
    }
}

pub fn build_softmax(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.extract_named_int("axis")?.unwrap_or(-1);
    Ok(Box::new(Softmax { axis }) as BoxOp)
}
