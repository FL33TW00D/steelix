use crate::prelude::*;
use std::{borrow::Cow, sync::Arc};

#[derive(Debug, Clone)]
pub struct Constant(pub Arc<Tensor>);

impl Op for Constant {
    fn name(&self) -> Cow<str> {
        "Constant".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Constant
    }

    fn realize(&self, _: PVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp {
            cost: OpCost {
                parameters: self.0.numel(),
                ..OpCost::default()
            },
            outputs: pvec![self.0.clone()],
        })
    }
}

pub fn build_constant(t: Tensor) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Constant(t.into_arc_tensor())) as BoxOp)
}
