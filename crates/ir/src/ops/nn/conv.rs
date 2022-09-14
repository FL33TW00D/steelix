use anyhow::bail;
use onnx::onnx_pb;
use std::borrow::Cow;

use crate::{BoxOp, Op, OpCost, OpGroup, QuadVec, RealizedOp};

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
    fn output_dims(&self, input_shape: &[i64]) -> (usize, usize) {
        let kernel_shape = self.kernel_shape.clone().unwrap();
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
}

impl Op for Conv {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Layer
    }

    fn cost(&self, providers: crate::QuadVec) -> anyhow::Result<RealizedOp> {
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

        Ok(RealizedOp {
            cost: OpCost {
                mac,
                parameters,
                flops: mac * 2,
            },
            outputs: QuadVec::new(),
        })
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
