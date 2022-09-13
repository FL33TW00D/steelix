use std::{borrow::Cow, sync::Arc};

use crate::{BoxOp, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Constant(pub Arc<Tensor>);

impl Op for Constant {
    fn name(&self) -> Cow<str> {
        "Constant".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Constant
    }

    fn realize(&self, _: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        Ok(vec![Arc::clone(&self.0)])
    }
}

//We need an OpBuilder trait
pub fn build_constant(t: Tensor) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Constant(t.into_arc_tensor())) as BoxOp)
}
