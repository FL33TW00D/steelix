use std::{borrow::Cow, sync::Arc};

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
    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        let mut qv = PVec::new();
        qv.push(self.0.clone());
        Ok(RealizedOp {
            cost: OpCost {
                flops: 0,
                parameters: 0,
            },
            outputs: qv,
        })
    }
}

pub fn build_constant(t: Tensor) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Constant(t.into_arc_tensor())) as BoxOp)
}
