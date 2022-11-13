use std::borrow::Cow;

use crate::prelude::*;
use steelix_onnx::onnx_pb;

#[derive(Debug, Clone)]
pub struct Unimplemented {
    pub op_type: Cow<'static, str>,
}

impl Op for Unimplemented {
    fn name(&self) -> Cow<str> {
        self.op_type.clone()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Unimplemented
    }
    fn realize(&self, _: PVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp::default())
    }
}

impl Unimplemented {}

pub fn build_unimplemented(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    println!(
        "Unimplemented operation found: {:?}. Try plotting the model with `--disable-shapes`",
        proto.op_type
    );
    Ok(Box::new(Unimplemented {
        op_type: proto.op_type.clone().into(),
    }) as BoxOp)
}
