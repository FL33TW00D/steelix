use std::{borrow::Cow, sync::Arc};

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Unimplemented;

impl Op for Unimplemented {
    fn name(&self) -> Cow<str> {
        "Unimplemented".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Unimplemented
    }

    fn realize(&self, _: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        todo!()
    }
}

pub fn build_unimplemented(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Unimplemented) as BoxOp)
}
