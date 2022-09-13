use onnx::onnx_pb;
use std::{borrow::Cow, ops::AddAssign, sync::Arc};

use crate::{
    as_float, ops::shape::Pad, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor,
};

use super::Depthwise;

#[derive(Debug, Clone, Default)]
pub struct Conv {
    pub group: i64,
    pub pads: Vec<i64>,
    pub kernel_shape: Option<Vec<i64>>,
    pub strides: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl Conv {
    pub fn convolve<T: DataType + ndarray::LinalgScalar + AddAssign>(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>> {
        let k0 = self.kernel_shape.clone().unwrap()[0] as usize;
        let k1 = self.kernel_shape.clone().unwrap()[1] as usize;

        let s0 = self.strides.clone()[0] as usize;
        let s1 = self.strides.clone()[1] as usize;

        let (h_out, w_out) = Self::output_dims(
            self,
            &*input
                .shape
                .clone()
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>(),
        );
        let num_filters = kernel.shape[0];
        let input_channels = kernel.shape[1];

        let mut output = Tensor::zeros::<T>(vec![
            input.shape[0],
            num_filters,
            h_out as usize,
            w_out as usize,
        ]);

        let padded = Pad::pad::<T>(input, self.pads.clone()).unwrap();
        let pad_m = padded.shape[2];
        let pad_n = padded.shape[3];

        let kptr = kernel.as_ptr::<T>().unwrap();
        let bptr = bias.as_ptr::<T>().unwrap();
        let iptr = padded.as_ptr::<T>().unwrap();
        let optr = output.as_mut_ptr::<T>().unwrap();

        for bn in 0..input.shape[0] {
            let input_batch_offset = bn * input_channels * pad_m * pad_n;
            let output_batch_offset = bn * num_filters * h_out * w_out;
            for filter in 0..num_filters {
                let output_filter_offset = filter * h_out * w_out;
                let kernel_filter_offset = filter * input_channels * k0 * k1;
                for channel in 0..input_channels {
                    let input_channel_offset = channel * pad_m * pad_n;
                    let kernel_channel_offset = channel * k0 * k1;
                    for row in 0..h_out {
                        let row_offset = row * s0;
                        for col in 0..w_out {
                            let col_offset = col * s1;
                            let mut sum = T::zero();
                            for kr in 0..k0 {
                                for kc in 0..k1 {
                                    sum += Tensor::get_value::<T>(
                                        &padded,
                                        &iptr,
                                        Self::input_index(
                                            row_offset,
                                            kr,
                                            col_offset,
                                            kc,
                                            pad_n,
                                            input_batch_offset,
                                            input_channel_offset,
                                        ),
                                    )
                                    .unwrap()
                                        * Tensor::get_value::<T>(
                                            kernel,
                                            &kptr,
                                            Self::kernel_index(
                                                kr,
                                                kc,
                                                k0,
                                                kernel_filter_offset,
                                                kernel_channel_offset,
                                            ),
                                        )
                                        .unwrap();
                                }
                            }

                            let oo = Self::output_index(
                                row,
                                col,
                                h_out,
                                output_batch_offset,
                                output_filter_offset,
                            );

                            unsafe {
                                *optr.add(oo) += sum;

                                if channel == 0 {
                                    *optr.add(oo) += *bptr.add(filter);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(output.into_arc_tensor())
    }

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

    #[inline]
    fn input_index(
        row: usize,
        kr: usize,
        col: usize,
        kc: usize,
        n: usize,
        batch_offset: usize,
        channel_offset: usize,
    ) -> usize {
        batch_offset + channel_offset + (row * n) + col + (kr * n) + kc
    }

    #[inline]
    fn output_index(
        row: usize,
        col: usize,
        h_out: usize,
        batch_offset: usize,
        filter_offset: usize,
    ) -> usize {
        batch_offset + filter_offset + (row * h_out) + col
    }

    #[inline]
    fn kernel_index(
        kr: usize,
        kc: usize,
        k0: usize,
        filter_offset: usize,
        channel_offset: usize,
    ) -> usize {
        filter_offset + channel_offset + (kr * k0) + kc
    }
}

impl Op for Conv {
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

        if providers.len() == 2 {
            Ok(vec![as_float!(Conv::convolve(providers[0].dt)(
                self,
                &providers[0],
                &providers[1],
                &as_float!(Tensor::zeros(providers[0].dt)(vec![providers[1].shape[1]]))
            ))?])
        } else {
            Ok(vec![as_float!(Conv::convolve(providers[0].dt)(
                self,
                &providers[0],
                &providers[1],
                &providers[2]
            ))?])
        }
    }
}

pub fn build_conv(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
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
        Ok(Box::new(Conv {
            group,
            pads,
            kernel_shape,
            strides,
            dilations,
        }) as BoxOp)
    }
}
