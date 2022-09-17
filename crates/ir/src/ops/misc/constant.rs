use std::{borrow::Cow, sync::Arc};

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, QuadVec, RealizedOp, Tensor};

#[derive(Debug, Clone)]
pub struct Constant(pub Arc<Tensor>);

impl Op for Constant {
    fn name(&self) -> Cow<str> {
        "Constant".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Constant
    }
    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        let mut qv = QuadVec::new();
        qv.push(self.0.clone());
        Ok(RealizedOp {
            cost: OpCost {
                mac: 0,
                parameters: 0,
            },
            outputs: qv,
        })
    }
}

pub fn build_constant(t: Tensor) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Constant(t.into_arc_tensor())) as BoxOp)
}
