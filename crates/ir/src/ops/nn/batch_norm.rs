use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{validate_providers, BoxOp, Op, OpCost, OpGroup, PVec, RealizedOp};

use smallvec::smallvec;

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
        let mac = providers[0].numel();
        let parameters = providers[1..4]
            .iter()
            .fold(0, |total, current| total + current.numel());
        Ok(RealizedOp {
            cost: OpCost {
                flops: mac,
                parameters,
            },
            outputs: smallvec![providers[0].clone(); 4],
        })
    }

    fn update(&mut self, _t: std::sync::Arc<crate::Tensor>) {}
}

pub fn build_batchnorm(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let epsilon = proto.get_attribute("epsilon", Some(1e-5), proto)?;
    Ok(Box::new(BatchNormalization { epsilon }) as BoxOp)
}
