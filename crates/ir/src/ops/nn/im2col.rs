use ndarray::{Array, Array2, Ix2};
use onnx::onnx_pb;
use std::{borrow::Cow, ops::AddAssign, sync::Arc};

use crate::{
    as_float,
    ops::shape::{im2col, Pad},
    BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor,
};

use super::{Depthwise};

#[derive(Debug, Clone, Default)]
pub struct Im2Col {
    pub group: i64,
    pub pads: Vec<i64>,
    pub kernel_shape: Option<Vec<i64>>,
    pub strides: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl Im2Col {
    ///Algorithm as 2 main steps:
    /// 1. Transform input samples from (N,C,H,W) to (N,I2CH, I2CW)
    /// 2. GEMM with Kernel
    ///Output: N, OC, OH, OW
    pub fn convolve<T: DataType + ndarray::LinalgScalar + AddAssign>(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
    ) -> anyhow::Result<Arc<Tensor>> {
        let kernel_shape = &kernel.shape;
        let (kn, kch, kr, kc) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );
        let (s0, s1) = (self.strides[0] as usize, self.strides[1] as usize);

        let padded = Pad::pad::<T>(input, self.pads.clone()).unwrap();
        let input_shape = &padded.shape;
        let n = input_shape[0];

        let (h_out, w_out) = Self::output_dims(
            self,
            &*input
                .shape
                .clone()
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>(),
        );

        let reshaped_kernel = kernel
            .to_array_view::<T>()
            .unwrap()
            .into_shape((kn, kr * kc * kch))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();

        let cols = im2col(&padded, kernel_shape, (s0, s1), (h_out, w_out));
        let mut convolved: Tensor = reshaped_kernel.dot(&cols).into();

        if let Some(bt) = bias {
            let cptr = convolved.as_mut_ptr::<T>().unwrap();
            let bptr = bt.as_ptr::<T>().unwrap();
            for o in 0..convolved.len {
                unsafe {
                    *cptr.add(o) += *bptr.add(o / (convolved.shape[1]));
                }
            }
        }
        let res = convolved
            .to_array_view::<T>()
            .unwrap()
            .to_owned()
            .into_shape((n, kn, h_out, w_out))
            .unwrap()
            .into_arc_tensor();

        Ok(res)
    }

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

    ///X: Previous layer input, (N x C x H x W)
    ///W: weight input
    ///Option<B>: bias
    ///
    ///     | i + 2p - k - (k - 1)(d - 1) |
    /// o = | --------------------------- |
    ///     |_             s             _|
    ///     
    /// size formula: https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    ///
    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        if providers.len() > 3 || providers.len() < 2 {
            panic!("Convolution had invalid number of inputs.")
        }

        let bias = if providers.len() == 3 {
            Some(providers[2].as_ref())
        } else {
            None
        };

        Ok(vec![as_float!(Im2Col::convolve(providers[0].dt)(
            self,
            &providers[0],
            &providers[1],
            bias
        ))?])
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
