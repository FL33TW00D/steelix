use std::{borrow::Cow, sync::Arc};

use num::cast::AsPrimitive;
use onnx::onnx_pb;

use crate::{as_float, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

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
}

pub fn build_batchnorm(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let epsilon = proto.extract_named_float("epsilon")?.unwrap_or(1e-5);
    Ok(Box::new(BatchNormalization { epsilon }) as BoxOp)
}
