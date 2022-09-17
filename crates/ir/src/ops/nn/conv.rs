use anyhow::bail;
use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{
    validate_providers, BoxOp, IntoArcTensor, Op, OpCost, OpGroup, QuadVec, RealizedOp, Tensor,
};

use super::Depthwise;

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
            - self.dilations[0] * (self.kernel_shape.clone()[0] - 1)
            - 1)
            / self.strides[0])
            + 1) as f32)
            .floor();

        let out_width = ((((input_shape[3] + (2 * self.pads[3])
            - self.dilations[1] * (self.kernel_shape.clone()[1] - 1)
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

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 3, self.name().to_string())?;
        if providers.len() > 3 || providers.len() < 2 {
            bail!("Conv providers incorrect length: {:?}", providers.len())
        }
        let x = providers[0].clone();
        let (n, cin, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);

        let w = providers[1].clone();
        let (f, kc, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);

        let (h_out, w_out) = self.output_dims(
            &x.shape
                .iter()
                .cloned()
                .map(|x| x as i64)
                .collect::<Vec<_>>(),
        );

        let mac = (cin / self.group as usize) * kh * kw * h_out * w_out * f;
        let parameters = f * cin * kh * (kw / self.group as usize);

        let placeholder = Tensor::zeros::<f32>(vec![n, f, h_out, w_out]).into_arc_tensor();

        Ok(RealizedOp {
            cost: OpCost { mac, parameters },
            outputs: smallvec![placeholder; 4],
        })
    }
}

pub fn build_conv(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let group = proto.get_attribute("group", Some(1), proto)?;
    let pads = proto.get_attribute("pads", Some(vec![0, 0, 0, 0]), proto)?;
    let kernel_shape = proto.get_attribute("kernel_shape", None, proto)?;
    let strides = proto.get_attribute("strides", None, proto)?;
    let dilations = proto.get_attribute("dilations", None, proto)?;

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
