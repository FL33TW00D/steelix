use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{
    validate_providers, BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp, Tensor,
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
        let placeholder =
            Tensor::new(providers[0].dt, providers[0].shape.clone()).into_arc_tensor();

        Ok(RealizedOp {
            cost: OpCost::zero_cost(),
            outputs: smallvec![placeholder],
        })
    }
}

pub fn build_dropout(_: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Dropout) as BoxOp)
}
