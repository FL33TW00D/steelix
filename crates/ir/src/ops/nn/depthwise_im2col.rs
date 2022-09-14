use std::{borrow::Cow, ops::AddAssign, sync::Arc};

use ndarray::{Array, Array2, Ix2};

use crate::{as_float, ops::shape::Pad, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone, Default)]
pub struct DepthwiseIm2col {
    pub group: i64,
    pub pads: Vec<i64>,
    pub kernel_shape: Option<Vec<i64>>,
    pub strides: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl DepthwiseIm2col {
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

impl Op for DepthwiseIm2col {
    fn name(&self) -> Cow<str> {
        "Depthwise".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Layer
    }
}
