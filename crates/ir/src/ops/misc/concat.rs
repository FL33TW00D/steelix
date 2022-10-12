use std::borrow::Cow;

use crate::{Op, OpGroup, PVec, RealizedOp};

#[derive(Debug, Clone)]
pub struct Concat;

impl Op for Concat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Tensor
    }
    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp::default())
    }
}
