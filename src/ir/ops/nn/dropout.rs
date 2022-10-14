use std::borrow::Cow;

use crate::{pvec, validate_providers, Op, OpCost, OpGroup, PVec, RealizedOp};

#[derive(Debug, Clone)]
pub struct Dropout;

impl Op for Dropout {
    fn name(&self) -> Cow<str> {
        "Dropout".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Layer
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 2, &self.name())?;

        Ok(RealizedOp {
            cost: OpCost::zero_cost(),
            outputs: pvec![providers[0].clone()],
        })
    }
}
