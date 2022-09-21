use onnx::onnx_pb;
use smallvec::{smallvec, SmallVec};
use std::borrow::Cow;

use crate::{
    as_std, validate_providers, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, PVec,
    RealizedOp, Shape, Tensor,
};
#[derive(Debug, Clone)]
pub struct concat {
    allow_zero: i64,
}

impl concat {
    pub fn concat<D: DataType + ndarray::LinalgScalar + num::NumCast>(
        shape_tensor: &Tensor,
    ) -> Shape {
        let data: Vec<D> = shape_tensor.as_slice().unwrap().into();
        let mut new_shape = SmallVec::new();
        for elem in data {
            new_shape.push(num::cast(elem).unwrap());
        }

        new_shape
    }
}

impl Op for concat {
    fn name(&self) -> Cow<str> {
        "concat".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 2, &self.name())?;

        let new_shape = as_std!(concat::concat(providers[0].dt)(&providers[1]));

        let concatd = Tensor::new(providers[0].dt, new_shape, None).into_arc_tensor();

        Ok(RealizedOp::zero_cost(smallvec![concatd]))
    }
}

pub fn build_concat(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let allow_zero = proto.get_attribute("allowzero", Some(0))?;
    Ok(Box::new(concat { allow_zero }) as BoxOp)
}
