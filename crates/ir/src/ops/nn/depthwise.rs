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
    ///Depthwise
    ///input tensor of shape (B, IC, IH, IW)
    ///kenel tensor of shape (F, IC, K0, K1)
    pub fn convolve<T: DataType + ndarray::LinalgScalar + AddAssign>(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
    ) -> anyhow::Result<Arc<Tensor>> {
        let kernel_shape = self.kernel_shape.clone().unwrap();
        let (k0, k1) = (kernel_shape[0] as usize, kernel_shape[1] as usize);

        let strides = self.strides.clone();
        let (s0, s1) = (strides[0] as usize, strides[1] as usize);

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
        let input_channels = input.shape[1];

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
        let mut bptr = None;
        if let Some(bt) = bias {
            bptr = Some(bt.as_ptr::<T>().unwrap());
        }
        let iptr = padded.as_ptr::<T>().unwrap();
        let optr = output.as_mut_ptr::<T>().unwrap();

        for bn in 0..input.shape[0] {
            let input_batch_offset = bn * input_channels * pad_m * pad_n;
            let output_batch_offset = bn * num_filters * h_out * w_out;
            for channel in 0..input_channels {
                let input_channel_offset = channel * pad_m * pad_n;
                let ouput_channel_offset = channel * h_out * w_out;
                let kernel_channel_offset = channel * k0 * k1;
                for row in 0..h_out {
                    let inprow = row * s0;
                    for col in 0..w_out {
                        let inpcol = col * s1;
                        let mut sum = if let Some(bp) = bptr {
                            unsafe { *bp.add(channel) }
                        } else {
                            T::zero()
                        };
                        for kr in 0..k0 {
                            for kc in 0..k1 {
                                unsafe {
                                    sum += *iptr.add(Self::input_index(
                                        inprow,
                                        kr,
                                        inpcol,
                                        kc,
                                        pad_n,
                                        input_batch_offset,
                                        input_channel_offset,
                                    )) * *kptr.add(Self::kernel_index(
                                        kr,
                                        kc,
                                        k0,
                                        kernel_channel_offset,
                                    ));
                                }
                            }
                        }
                        unsafe {
                            *optr.add(Self::output_index(
                                row,
                                col,
                                h_out,
                                output_batch_offset,
                                ouput_channel_offset,
                            )) += sum;
                        }
                    }
                }
            }
        }

        Ok(output.into_arc_tensor())
    }

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
        channel_offset: usize,
    ) -> usize {
        batch_offset + channel_offset + (row * h_out) + col
    }

    #[inline]
    fn kernel_index(kr: usize, kc: usize, k0: usize, channel_offset: usize) -> usize {
        channel_offset + (kr * k0) + kc
    }
}

impl Op for Depthwise {
    fn name(&self) -> Cow<str> {
        "Depthwise".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Layer
    }

    ///X: Previous layer input, (N x C x H x W)
    ///W: weight input
    ///Option<B>: bias
    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        if providers.len() > 3 || providers.len() < 2 {
            panic!("Depthwiseolution had invalid number of inputs.")
        }

        if providers.len() == 2 {
            Ok(vec![as_float!(Depthwise::convolve(providers[0].dt)(
                self,
                &providers[0],
                &providers[1],
                None
            ))?])
        } else {
            Ok(vec![as_float!(Depthwise::convolve(providers[0].dt)(
                self,
                &providers[0],
                &providers[1],
                Some(&providers[2])
            ))?])
        }
    }
}
