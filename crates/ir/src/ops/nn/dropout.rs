use std::borrow::Cow;

use crate::{
    pvec, validate_providers, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp, Tensor,
};

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
        let placeholder = Tensor::new(providers[0].dt, providers[0].shape.clone());

        Ok(RealizedOp {
            cost: OpCost::zero_cost(),
            outputs: pvec![placeholder.into_arc_tensor()],
        })
    }
}
