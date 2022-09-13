use std::{borrow::Cow, sync::Arc};

use num::FromPrimitive;
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

impl AvgPool {
    //avgpool(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    pub fn pool<T: DataType + ndarray::LinalgScalar + num_traits::FromPrimitive>(
        &self,
        input: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>> {
        let input_channels = input.shape[1];
        let m = input.shape[2] as i64;
        let n = input.shape[3] as i64;

        let k0 = self.kernel_shape.clone()[0];
        let k1 = self.kernel_shape.clone()[1];

        let s0 = self.strides.clone()[0];
        let s1 = self.strides.clone()[1];

        let p0 = self.pads.clone()[2];
        let p1 = self.pads.clone()[3];
        let h_out = ((((m + (2 * p0) - k0) as i64 / s0) + 1) as f32).floor() as i64;
        let w_out = ((((n + (2 * p1) - k1) as i64 / s1) + 1) as f32).floor() as i64;
        let padded = Pad::pad::<T>(input, self.pads.clone()).unwrap();
        let pad_m = padded.shape[2] as i64;
        let pad_n = padded.shape[3] as i64;
        let mut output = Tensor::zeros::<T>(vec![
            input.shape[0],
            input_channels,
            h_out as usize,
            w_out as usize,
        ]);

        let iptr = padded.as_ptr::<T>().unwrap();
        let optr = output.as_mut_ptr::<T>().unwrap();
        let filter_val = FromPrimitive::from_f32(1. / (k0 * k1) as f32).unwrap();

        for bn in 0..input.shape[0] {
            for channel in 0..input_channels {
                for row in 0..h_out {
                    for col in 0..w_out {
                        let mut sum = T::zero();
                        for kr in 0..k0 {
                            for kc in 0..k1 {
                                unsafe {
                                    let ii = Self::input_index(
                                        channel as i64,
                                        row * s0,
                                        kr,
                                        col * s1,
                                        kc,
                                        pad_m,
                                        pad_n,
                                        input_channels as i64,
                                        bn as i64,
                                    );
                                    let ival = *iptr.add(ii);
                                    sum = sum + ival * filter_val;
                                }
                            }
                        }
                        let oo = Self::output_index(
                            channel as i64,
                            row,
                            col,
                            h_out,
                            w_out,
                            input_channels as i64,
                            bn as i64,
                        );

                        unsafe {
                            *optr.add(oo) = *optr.add(oo) + sum;
                        }
                    }
                }
            }
        }

        Ok(output.into_arc_tensor())
    }

    #[inline]
    fn input_index(
        channel: i64,
        row: i64,
        kr: i64,
        col: i64,
        kc: i64,
        m: i64,
        n: i64,
        num_channels: i64,
        bn: i64,
    ) -> usize {
        let batch_offset = bn * num_channels * m * n;
        let channel_offset = channel * m * n;
        let inp_offset = (row * n) + col;
        let ker_offset = (kr * n) + kc;
        let ii = (batch_offset + (channel_offset + inp_offset + ker_offset)) as usize;
        ii
    }

    #[inline]
    fn output_index(
        channel: i64,
        row: i64,
        col: i64,
        h_out: i64,
        w_out: i64,
        num_channels: i64,
        bn: i64,
    ) -> usize {
        let batch_offset = bn * num_channels * h_out * w_out;
        let channel_offset = channel * h_out * w_out;
        let oo = (batch_offset + channel_offset + (row * h_out) + col) as usize;
        oo
    }
}

impl Op for AvgPool {
    fn name(&self) -> Cow<str> {
        "AveragePool".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Pool
    }

    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        Ok(vec![as_std!(AvgPool::pool(providers[0].dt)(
            self,
            &providers[0]
        ))?])
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
