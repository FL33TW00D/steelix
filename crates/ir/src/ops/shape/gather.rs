use bytes::BytesMut;
use onnx::onnx_pb;
use smallvec::{smallvec, SmallVec};
use std::borrow::Cow;

use crate::{
    as_std, validate_providers, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, PVec,
    RealizedOp, Shape, Tensor,
};
#[derive(Debug, Clone)]
pub struct Gather {
    pub axis: i64,
}

impl Gather {
    pub fn compute_output_shape<D: num::NumCast + std::clone::Clone>(
        &self,
        input_shape: &[D],
        indices_shape: &[D],
    ) -> anyhow::Result<Shape> {
        let mut output_shape = smallvec![];
        for (idx, dim) in input_shape.iter().enumerate() {
            if idx as i64 != self.axis {
                output_shape.push(num::cast((*dim).clone()).unwrap());
            } else {
                for idx2 in indices_shape {
                    output_shape.push(num::cast((*idx2).clone()).unwrap());
                }
            }
        }
        Ok(output_shape)
    }
}

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 2, &self.name())?;

        let input_shape = &providers[0].shape;
        let indices_shape = &providers[1].shape;
        let output_shape = self.compute_output_shape(&input_shape, &indices_shape)?;

        let out = Tensor::new(providers[0].dt, output_shape.into(), None);
        Ok(RealizedOp::zero_cost(smallvec![out.into_arc_tensor()]))
    }
}

pub fn build_gather(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.get_attribute("axis", Some(0))?;
    Ok(Box::new(Gather { axis }) as BoxOp)
}
