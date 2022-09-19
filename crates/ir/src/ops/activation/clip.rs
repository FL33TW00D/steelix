use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{validate_providers, BoxOp, Op, OpCost, OpGroup, PVec, RealizedOp};

#[derive(Debug, Clone)]
pub struct Clip {
    pub min: i64,
    pub max: i64,
}

impl Op for Clip {
    fn name(&self) -> Cow<str> {
        "Clip".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Activation
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 3, &self.name())?;
        let mut qv = PVec::new();
        qv.push(providers[0].clone());

        Ok(RealizedOp {
            cost: OpCost {
                flops: providers[0].numel(),
                parameters: 0,
            },
            outputs: qv,
        })
    }
}

pub fn build_clip(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let min = proto.get_attribute("min", Some(i64::MIN))?;
    let max = proto.get_attribute("max", Some(i64::MAX))?;
    Ok(Box::new(Clip { min, max }) as BoxOp)
}
