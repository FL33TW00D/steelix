use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, RealizedOp, Tensor};

use smallvec::smallvec;

#[derive(Debug, Clone)]
pub struct AvgPool {
    pub ceil_mode: Option<i64>,
    pub count_include_pad: Option<i64>,
    pub pads: Vec<i64>,
    pub strides: Vec<i64>,
    pub kernel_shape: Vec<i64>,
}

impl AvgPool {
    fn output_dims(&self, m: i64, n: i64) -> (usize, usize) {
        let k0 = self.kernel_shape.clone()[0];
        let k1 = self.kernel_shape.clone()[1];

        let s0 = self.strides.clone()[0];
        let s1 = self.strides.clone()[1];

        let p0 = self.pads.clone()[2];
        let p1 = self.pads.clone()[3];
        let h_out = ((((m + (2 * p0) - k0) as i64 / s0) + 1) as f32).floor() as usize;
        let w_out = ((((n + (2 * p1) - k1) as i64 / s1) + 1) as f32).floor() as usize;
        (h_out, w_out)
    }
}

impl Op for AvgPool {
    fn name(&self) -> Cow<str> {
        "AveragePool".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Pool
    }

    fn cost(&self, providers: crate::QuadVec) -> anyhow::Result<crate::RealizedOp> {
        let input_shape = &providers[0].shape;
        let (h_out, w_out) = self.output_dims(input_shape[2] as i64, input_shape[3] as i64);
        let out_shape = vec![input_shape[0], input_shape[1], h_out, w_out];
        let out = Tensor::zeros::<f32>(out_shape);
        Ok(RealizedOp {
            cost: OpCost {
                mac: providers[0].numel(),
                parameters: 0,
            },
            outputs: smallvec![out.into_arc_tensor(); 4],
        })
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
