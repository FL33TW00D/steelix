use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{validate_providers, BoxOp, Op, OpCost, OpGroup, PVec, RealizedOp};

use smallvec::smallvec;

#[derive(Debug, Clone)]
pub struct LRN {
    pub alpha: f32,
    pub beta: f32,
    pub bias: f32,
    pub size: i64,
}

impl Op for LRN {
    fn name(&self) -> Cow<str> {
        "LRN".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Normalization
    }

    //[gamma weights, beta weights, moving_mean(non-trainable), moving_variance(non-trainable)]
    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 1, &self.name())?;
        let mac = providers[0].numel() * 4;
        Ok(RealizedOp {
            cost: OpCost {
                flops: mac,
                parameters: 0,
            },
            outputs: smallvec![providers[0].clone(); 4],
        })
    }
}

pub fn build_lrn(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let alpha = proto.get_attribute("alpha", Some(1e-4), proto)?;
    let beta = proto.get_attribute("beta", Some(7.5e-1), proto)?;
    let bias = proto.get_attribute("bias", Some(1.), proto)?;
    let size = proto.get_attribute("size", None, proto)?;
    Ok(Box::new(LRN {
        alpha,
        beta,
        bias,
        size,
    }) as BoxOp)
}
