use smallvec::smallvec;
use std::{borrow::Cow, sync::Arc};

use crate::{BoxOp, IntoArcTensor, Op, OpGroup, PVec, RealizedOp, Tensor, ValueInfo};

#[derive(Debug, Clone)]
pub struct Initial(Arc<Tensor>);

impl Op for Initial {
    fn name(&self) -> Cow<str> {
        "Initial".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Data
    }

    fn realize(&self, _: PVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp::zero_cost(smallvec![self.0.clone(); 4]))
    }
}

pub fn build_initial(value_info: ValueInfo) -> Result<BoxOp, anyhow::Error> {
    let initial = Tensor::new(crate::DType::F32, value_info.dimensions);
    Ok(Box::new(Initial(initial.into_arc_tensor())) as BoxOp)
}
