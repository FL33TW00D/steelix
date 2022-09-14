use std::{borrow::Cow, ops::AddAssign, sync::Arc};

use crate::{as_float, ops::shape::Pad, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone, Default)]
pub struct Depthwise {
    pub group: i64,
    pub pads: Vec<i64>,
    pub kernel_shape: Option<Vec<i64>>,
    pub strides: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl Depthwise {
    ///     | i + 2p - k - (k - 1)(d - 1) |
    /// o = | --------------------------- |
    ///     |_             s             _|
    fn output_dims(&self, input_shape: &[i64]) -> (usize, usize) {
        let out_height = ((((input_shape[2] + (2 * self.pads[2])
            - self.dilations[0] * (self.kernel_shape.clone().unwrap()[0] - 1)
            - 1)
            / self.strides[0])
            + 1) as f32)
            .floor();

        let out_width = ((((input_shape[3] + (2 * self.pads[3])
            - self.dilations[1] * (self.kernel_shape.clone().unwrap()[1] - 1)
            - 1)
            / self.strides[1])
            + 1) as f32)
            .floor();

        (out_height as usize, out_width as usize)
    }
}

impl Op for Depthwise {
    fn name(&self) -> Cow<str> {
        "Depthwise".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Layer
    }
}
