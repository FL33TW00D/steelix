use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, QuadVec, RealizedOp, Tensor};

#[derive(Debug, Clone, Default)]
pub struct Im2Col {
    pub group: i64,
    pub pads: Vec<i64>,
    pub kernel_shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl Im2Col {
    fn output_dims(&self, input_shape: &[i64]) -> (usize, usize) {
        let kernel_shape = self.kernel_shape.clone();
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

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
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

pub fn build_im2col(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let group = proto.get_attribute("group", Some(1), proto)?;
    let pads = proto.get_attribute("pads", Some(vec![0, 0, 0, 0]), proto)?;
    let kernel_shape = proto.get_attribute("kernel_shape", None, proto)?;
    let strides = proto.get_attribute("strides", None, proto)?;
    let dilations = proto.get_attribute("dilations", None, proto)?;

    Ok(Box::new(Im2Col {
        group,
        pads,
        kernel_shape,
        strides,
        dilations,
    }) as BoxOp)
}
