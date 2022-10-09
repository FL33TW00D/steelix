use onnx::onnx_pb;
use smallvec::{smallvec, SmallVec};
use std::borrow::Cow;

use crate::{
    as_std, validate_providers, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, PVec,
    RealizedOp, Shape, Tensor,
};
#[derive(Debug, Clone)]
pub struct Reshape {
    pub allow_zero: i64,
}

impl Reshape {
    pub fn reshape<D: DataType + ndarray::LinalgScalar + num::NumCast>(
        shape_tensor: &Tensor,
    ) -> Shape {
        let data: Vec<D> = shape_tensor.as_slice().unwrap().into();
        let mut new_shape = SmallVec::new();
        for elem in data {
            new_shape.push(num::cast(elem).unwrap_or_else(|| panic!("Failed to cast: {:?}", elem)));
        }

        new_shape
    }
}

impl Op for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        println!("Reshape providers: {:?}", providers);
        validate_providers(&providers, 2, 2, &self.name())?;
        let new_shape = as_std!(Reshape::reshape(providers[1].dt)(&providers[1]));

        let reshaped = Tensor::new(providers[0].dt, new_shape).into_arc_tensor();

        Ok(RealizedOp::zero_cost(smallvec![reshaped]))
    }
}

pub fn build_reshape(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let allow_zero = proto.get_attribute("allowzero", Some(0))?;
    Ok(Box::new(Reshape { allow_zero }) as BoxOp)
}
