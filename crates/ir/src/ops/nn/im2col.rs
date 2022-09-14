use ndarray::{Array, Array2, Ix2};
use onnx::onnx_pb;
use std::{borrow::Cow, ops::AddAssign, sync::Arc};

use crate::{
    as_float,
    ops::shape::{im2col, Pad},
    BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor,
};

use super::Depthwise;

#[derive(Debug, Clone, Default)]
pub struct Im2Col {
    pub group: i64,
    pub pads: Vec<i64>,
    pub kernel_shape: Option<Vec<i64>>,
    pub strides: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl Im2Col {
    fn output_dims(&self, input_shape: &[i64]) -> (usize, usize) {
        let kernel_shape = self.kernel_shape.clone().unwrap();
        let (kr, kc) = (kernel_shape[0], kernel_shape[1]);
        let out_height =
            ((((input_shape[2] + (2 * self.pads[2]) - kr) / self.strides[0]) + 1) as f32).floor();

        let out_width =
            ((((input_shape[3] + (2 * self.pads[3]) - kc) / self.strides[1]) + 1) as f32).floor();

        (out_height as usize, out_width as usize)
    }
}

impl Op for Im2Col {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Layer
    }
}

pub fn build_im2col(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let group = proto.extract_named_int("group")?.unwrap_or(1);
    let pads = proto
        .extract_named_intv("pads")?
        .unwrap_or_else(|| vec![0, 0, 0, 0]);
    let kernel_shape = proto.extract_named_intv("kernel_shape")?;
    let strides = proto.extract_named_intv("strides")?.unwrap();
    let dilations = proto.extract_named_intv("dilations")?.unwrap();

    if group != 1 {
        Ok(Box::new(Depthwise {
            group,
            pads,
            kernel_shape,
            strides,
            dilations,
        }) as BoxOp)
    } else {
        Ok(Box::new(Im2Col {
            group,
            pads,
            kernel_shape,
            strides,
            dilations,
        }) as BoxOp)
    }
}
