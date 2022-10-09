use std::{borrow::Cow, sync::Arc};

use smallvec::smallvec;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp, Tensor};

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
        println!(
            "Constant realize: {:?} {:?} {:?}",
            self.0.len, self.0.shape, self.0.dt
        );
        Ok(RealizedOp {
            cost: OpCost {
                flops: 0,
                parameters: self.0.numel(),
            },
            outputs: smallvec![self.0.clone()],
        })
    }
}

pub fn build_constant(t: Tensor) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Constant(t.into_arc_tensor())) as BoxOp)
}
