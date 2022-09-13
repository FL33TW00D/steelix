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
    ///Depthwise Im2col
    ///Depthwise im2col can be thought of as a GEMV
    ///input tensor of shape (B, IC, IH, IW)
    ///kenel tensor of shape (F, 1, K0, K1)
    ///Where IC==F
    ///
    ///We reshape the input tensor to (K0*K1, B*OH*OW)
    pub fn convolve<T: DataType + ndarray::LinalgScalar + AddAssign>(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>> {
        let kernel_shape = &kernel.shape;
        let (kn, kch, kr, kc) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );

        let input_shape = &input.shape;
        let n = input_shape[0];
        let _c = input_shape[1];

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

        let cols = Self::im2col::<T>(self, input, kernel_shape);

        let mut convolved: Tensor = reshaped_kernel.dot(&cols).into();

        let cptr = convolved.as_mut_ptr::<T>().unwrap();
        let bptr = bias.as_ptr::<T>().unwrap();

        for o in 0..convolved.len {
            let bidx = o / (convolved.shape[1]);

            if bidx > bias.len && o % 100 == 0 {
                println!("OOB: {:?} {:?}", bidx, bias.len);
            }

            unsafe {
                *cptr.add(o) += *bptr.add(bidx);
            }
        }

        Ok(convolved
            .to_array_view::<T>()
            .unwrap()
            .to_owned()
            .into_shape((n, kn, h_out, w_out))
            .unwrap()
            .into_arc_tensor())
    }

    pub fn im2col<T: DataType + ndarray::LinalgScalar>(
        &self,
        input: &Tensor,
        kernel_shape: &[usize],
    ) -> Array2<T> {
        let (kr, kc) = (kernel_shape[2], kernel_shape[3]);
        let s0 = self.strides[0] as usize;
        let s1 = self.strides[1] as usize;

        let padded = Pad::pad::<T>(input, self.pads.clone()).unwrap();
        let input_shape = &padded.shape;
        let (n, c, ih, iw) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let (h_out, w_out) = Self::output_dims(
            self,
            &*input.shape.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        );

        let mut output = Tensor::zeros::<T>(vec![kr * kc, n * h_out as usize * w_out as usize]);
        let iptr = padded.as_ptr::<T>().unwrap();
        let optr = output.as_mut_ptr::<T>().unwrap();
        let mut oidx = 0;
        for b in 0..n {
            let batch_offset = b * c * ih * iw;
            for h in (0..(ih - kr + 1)).step_by(s0) {
                for w in (0..(iw - kc + 1)).step_by(s1) {
                    let anchor = batch_offset + (h * ih) + w;
                    for channel in 0..c {
                        let channel_offset = channel * ih * iw;
                        for kri in 0..kr {
                            for kci in 0..kc {
                                let kernel_offset = (kri * iw) + kci;
                                let idx = anchor + channel_offset + kernel_offset;
                                unsafe {
                                    let ival = *iptr.add(idx);
                                    *optr.add(oidx) = ival;
                                }
                                oidx += 1;
                            }
                        }
                    }
                }
            }
        }
        let output_nd = output.to_array_view::<T>().unwrap();
        let output_transpose = output_nd.t();
        let mut transposed = Array::zeros(output_transpose.raw_dim());
        transposed.assign(&output_transpose);
        transposed.into_dimensionality::<Ix2>().unwrap()
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

impl Op for DepthwiseIm2col {
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
            panic!("Depthwise had invalid number of inputs.")
        }

        if providers.len() == 2 {
            Ok(vec![as_float!(
                DepthwiseIm2col::convolve(providers[0].dt)(
                    self,
                    &providers[0],
                    &providers[1],
                    &as_float!(Tensor::zeros(providers[0].dt)(vec![providers[1].shape[1]]))
                )
            )?])
        } else {
            Ok(vec![as_float!(
                DepthwiseIm2col::convolve(providers[0].dt)(
                    self,
                    &providers[0],
                    &providers[1],
                    &providers[2]
                )
            )?])
        }
    }
}
