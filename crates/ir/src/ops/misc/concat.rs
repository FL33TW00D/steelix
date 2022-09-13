use std::borrow::Cow;

use crate::{Op, OpGroup};

#[derive(Debug)]
pub struct Concat;

impl Op for Concat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Tensor
    }
}
