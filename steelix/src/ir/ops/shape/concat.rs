use crate::ir::OpError;
use crate::prelude::*;
use steelix_onnx::onnx_pb;
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct Concat {
    axis: i64,
}

impl Concat {
    pub fn concat(&self, providers: &PVec) -> Result<Shape, OpError> {
        Ok(Tensor::stack_tensors(self.axis as usize, providers)?.shape)
    }
}

impl Op for Concat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 2, &self.name())?;
        let new_shape = self.concat(&providers)?;

        Ok(RealizedOp::zero_cost(pvec!(Tensor::new(
            providers[0].dt,
            new_shape
        )
        .into_arc_tensor())))
    }
}

pub fn build_concat(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.get_attribute("axis", None)?;
    Ok(Box::new(Concat { axis }) as BoxOp)
}
