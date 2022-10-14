use smallvec::smallvec;
use std::{borrow::Cow, sync::Arc};

use crate::{BoxOp, DType, IntoArcTensor, Op, OpGroup, PVec, RealizedOp, Tensor, ValueInfo};

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
        Ok(RealizedOp::zero_cost(smallvec![self.0.clone()]))
    }
}

pub fn build_initial(value_info: ValueInfo) -> Result<BoxOp, anyhow::Error> {
    //TODO: this needs to be dynamic on DTYPE
    Ok(Box::new(Initial(
        Tensor::new(DType::F32, value_info.dimensions).into_arc_tensor(),
    )) as BoxOp)
}
