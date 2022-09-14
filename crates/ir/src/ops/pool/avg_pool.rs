use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{as_std, ops::shape::Pad, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct AvgPool {
    pub ceil_mode: Option<i64>,
    pub count_include_pad: Option<i64>,
    pub pads: Vec<i64>,
    pub strides: Vec<i64>,
    pub kernel_shape: Vec<i64>,
}

impl Op for AvgPool {
    fn name(&self) -> Cow<str> {
        "AveragePool".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Pool
    }
}

pub fn build_avgpool(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let ceil_mode = proto.extract_named_int("ceil_mode")?;
    let count_include_pad = proto.extract_named_int("count_include_pad")?;
    let pads = proto
        .extract_named_intv("pads")?
        .unwrap_or_else(|| vec![0, 0, 0, 0]);
    let kernel_shape = proto.extract_named_intv("kernel_shape")?.unwrap();
    let strides = proto
        .extract_named_intv("strides")?
        .unwrap_or_else(|| vec![1, 1]);
    Ok(Box::new(AvgPool {
        ceil_mode,
        count_include_pad,
        pads,
        strides,
        kernel_shape,
    }) as BoxOp)
}
