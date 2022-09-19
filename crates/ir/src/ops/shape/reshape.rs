use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{validate_providers, BoxOp, IntoArcTensor, Op, OpGroup, PVec, RealizedOp, Tensor};
#[derive(Debug, Clone)]
pub struct Reshape {
    allow_zero: i64,
}

impl Reshape {}

impl Op for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 2, &self.name())?;

        let shape_tensor = &providers[1].clone();

        Ok(RealizedOp::zero_cost(smallvec![Tensor::new(
            crate::DType::F32,
            shape_tensor.shape.clone()
        )
        .into_arc_tensor()]))
    }
}

pub fn build_reshape(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let allow_zero = proto.get_attribute("allowzero", Some(0))?;
    Ok(Box::new(Reshape { allow_zero }) as BoxOp)
}
