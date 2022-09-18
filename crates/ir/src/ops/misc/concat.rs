use std::borrow::Cow;

use crate::{Op, OpGroup, RealizedOp};

#[derive(Debug)]
pub struct Concat;

impl Op for Concat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Tensor
    }
    fn realize(&self, providers: crate::PVec) -> anyhow::Result<crate::RealizedOp> {
        Ok(RealizedOp::default())
    }
}
