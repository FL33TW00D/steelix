use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpGroup, RealizedOp};

#[derive(Debug, Clone)]
pub struct Squeeze {
    pub axes: Option<Vec<usize>>,
}

impl Op for Squeeze {
    fn name(&self) -> Cow<str> {
        "Squeeze".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn cost(&self, providers: crate::QuadVec) -> anyhow::Result<crate::RealizedOp> {
        Ok(RealizedOp::default())
    }
}

pub fn build_squeeze(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axes = proto.extract_named_attr("axes")?.unwrap();
    Ok(Box::new(Squeeze {
        axes: Some(axes.ints.clone().iter().map(|&i| i as usize).collect()),
    }) as BoxOp)
}
