use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, RealizedOp, Tensor};

use smallvec::smallvec;

#[derive(Debug, Clone)]
pub struct AvgPool {
    pub ceil_mode: i64,
    pub count_include_pad: i64,
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
    let ceil_mode = proto.get_attribute("ceil_mode", Some(0), proto)?;
    let count_include_pad = proto.get_attribute("count_include_pad", Some(0), proto)?;
    let pads = proto.get_attribute("pads", Some(vec![0, 0, 0, 0]), proto)?;
    let kernel_shape = proto.get_attribute("kernel_shape", None, proto)?; //TODO: fix
    let strides = proto.get_attribute("strides", Some(vec![1, 1]), proto)?;

    Ok(Box::new(AvgPool {
        ceil_mode,
        count_include_pad,
        pads,
        strides,
        kernel_shape,
    }) as BoxOp)
}
