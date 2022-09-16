use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, Op, OpGroup, QuadVec, RealizedOp};

#[derive(Debug, Clone)]
pub struct BatchNormalization {
    pub epsilon: f32,
}

impl Op for BatchNormalization {
    fn name(&self) -> Cow<str> {
        "BatchNormalization".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Normalization
    }

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp::default())
    }
}

pub fn build_batchnorm(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let epsilon = proto.extract_named_float("epsilon")?.unwrap_or(1e-5);
    Ok(Box::new(BatchNormalization { epsilon }) as BoxOp)
}
