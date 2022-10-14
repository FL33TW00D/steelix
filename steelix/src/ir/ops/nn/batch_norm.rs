use std::borrow::Cow;

use crate::prelude::*;
use steelix_onnx::onnx_pb;

#[derive(Debug, Clone)]
pub struct BatchNormalization {
    pub epsilon: f32,
}

impl Op for BatchNormalization {
    fn name(&self) -> Cow<str> {
        "BatchNormalization".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Normalization
    }

    //[gamma weights, beta weights, moving_mean(non-trainable), moving_variance(non-trainable)]
    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 5, 5, &self.name())?;
        Ok(RealizedOp {
            cost: OpCost {
                flops: providers[0].numel(),
                ..OpCost::default()
            },
            outputs: pvec![providers[0].clone()],
        })
    }
}

pub fn build_batchnorm(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let epsilon = proto.get_attribute("epsilon", Some(1e-5))?;
    Ok(Box::new(BatchNormalization { epsilon }) as BoxOp)
}
