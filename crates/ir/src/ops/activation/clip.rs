use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{validate_providers, BoxOp, Op, OpCost, OpGroup, QuadVec, RealizedOp};

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

    fn realize(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 3, self.name().to_string())?;
        let mut qv = QuadVec::new();
        qv.push(providers[0].clone());

        Ok(RealizedOp {
            cost: OpCost {
                mac: 1,
                parameters: 1000,
            },
            outputs: qv,
        })
    }
}

pub fn build_clip(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let min = proto.get_attribute("min", Some(i64::MIN), proto)?;
    let max = proto.get_attribute("max", Some(i64::MAX), proto)?;
    Ok(Box::new(Clip { min, max }) as BoxOp)
}
