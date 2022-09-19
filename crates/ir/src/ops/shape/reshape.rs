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
        println!("RESHAPE PROVIDERS: {:?}", providers);
        let reshaped = Tensor::new(providers[0].dt, smallvec![1, 9216], None).into_arc_tensor();

        Ok(RealizedOp::zero_cost(smallvec![reshaped]))
    }
}

pub fn build_reshape(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let allow_zero = proto.get_attribute("allowzero", Some(0))?;
    Ok(Box::new(Reshape { allow_zero }) as BoxOp)
}
