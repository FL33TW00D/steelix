use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpGroup};

#[derive(Debug, Clone)]
pub struct Clip {
    pub min: Option<i64>,
    pub max: Option<i64>,
}

impl Op for Clip {
    fn name(&self) -> Cow<str> {
        "Clip".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Activation
    }
}

pub fn build_clip(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let _min = proto.extract_named_attr("min")?;
    let _max = proto.extract_named_attr("max")?;
    Ok(Box::new(Clip {
        min: Some(0),
        max: Some(0),
    }) as BoxOp)
}
