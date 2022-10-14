use std::borrow::Cow;

use crate::prelude::*;
use onnx::onnx_pb;

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
        Ok(RealizedOp {
            cost: OpCost {
                flops: providers[0].numel(),
                ..OpCost::default()
            },
            outputs: pvec![providers[0].clone()],
        })
    }
}

pub fn build_lrn(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let alpha = proto.get_attribute("alpha", Some(1e-4))?;
    let beta = proto.get_attribute("beta", Some(7.5e-1))?;
    let bias = proto.get_attribute("bias", Some(1.))?;
    let size = proto.get_attribute("size", None)?;
    Ok(Box::new(LRN {
        alpha,
        beta,
        bias,
        size,
    }) as BoxOp)
}
