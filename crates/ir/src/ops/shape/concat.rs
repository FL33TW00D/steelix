use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{
    as_std, validate_providers, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, PVec,
    RealizedOp, Shape, Tensor,
};
#[derive(Debug, Clone)]
pub struct Concat {
    axis: i64,
}

impl Concat {
    pub fn concat<D: DataType + ndarray::LinalgScalar + num::NumCast>(providers: PVec) -> Shape {
        //Now what
        //
        Shape::new()
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

        println!("CONCAT PROVIDERS: {:?}", providers);
        let new_shape = as_std!(Concat::concat(providers[0].dt)(providers));

        Ok(RealizedOp::zero_cost(smallvec![]))
    }
}

pub fn build_concat(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.get_attribute("axis", None)?;
    Ok(Box::new(Concat { axis }) as BoxOp)
}
