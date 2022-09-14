use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpGroup};

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
}

pub fn build_softmax(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.extract_named_int("axis")?.unwrap_or(-1);
    Ok(Box::new(Softmax { axis }) as BoxOp)
}
