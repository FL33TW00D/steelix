use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{
    validate_providers, BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp, Tensor,
};

#[derive(Debug, Clone, Default)]
pub struct Conv {
    pub group: i64,
    pub pads: Vec<i64>,
    pub kernel_shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl Conv {
    fn output_dims(&self, input_shape: &[i64]) -> (usize, usize) {
        let kernel_shape = self.kernel_shape.clone();
        let out_height = ((((input_shape[2] + (2 * self.pads[2])
            - self.dilations[0] * (kernel_shape[0] - 1)
            - 1)
            / self.strides[0])
            + 1) as f32)
            .floor();

        let out_width = ((((input_shape[3] + (2 * self.pads[3])
            - self.dilations[1] * (kernel_shape[1] - 1)
            - 1)
            / self.strides[1])
            + 1) as f32)
            .floor();

        (out_height as usize, out_width as usize)
    }
}

impl Op for Conv {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Layer
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 3, &self.name())?;
        let x = providers[0].clone();
        let (n, cin, _, _) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);

        let w = providers[1].clone();
        let (f, _, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);

        let (h_out, w_out) = self.output_dims(
            &x.shape
                .iter()
                .cloned()
                .map(|x| x as i64)
                .collect::<Vec<_>>(),
        );

        let mac = (cin / self.group as usize) * kh * kw * h_out * w_out * f;
        let parameters = f * cin * kh * (kw / self.group as usize);

        let placeholder =
            Tensor::new(providers[0].dt, smallvec![n, f, h_out, w_out]).into_arc_tensor();

        Ok(RealizedOp {
            cost: OpCost {
                flops: mac * 2,
                parameters,
            },
            outputs: smallvec![placeholder],
        })
    }
}

pub fn build_conv(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let group = proto.get_attribute("group", Some(1))?;
    let pads = proto.get_attribute("pads", Some(vec![0, 0, 0, 0]))?;
    let kernel_shape = proto.get_attribute("kernel_shape", None)?;
    let strides = proto.get_attribute("strides", None)?;
    let dilations = proto.get_attribute("dilations", Some(vec![1, 1, 1, 1]))?;

    Ok(Box::new(Conv {
        group,
        pads,
        kernel_shape,
        strides,
        dilations,
    }) as BoxOp)
}
