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
    start: i64,
    end: i64,
}

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Gather
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 1, &self.name())?;
        let input_shape = &providers[0].shape;

        let end = if self.end == -1 {
            input_shape.len() as i64
        } else {
            self.end
        };

        let new_shape = &providers[0].shape[self.start as usize..end as usize];

        let bytes: Vec<u8> = new_shape.iter().flat_map(|s| s.to_ne_bytes()).collect();

        let out = Tensor::new(
            providers[0].dt,
            smallvec![new_shape.len()],
            Some((*bytes).into()),
        );

        Ok(RealizedOp::zero_cost(smallvec![out.into_arc_tensor()]))
    }
}

pub fn build_shape(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let start = proto.get_attribute("start", Some(0))? as i64;
    let end = proto.get_attribute("end", Some(-1))? as i64;
    Ok(Box::new(Gather { start, end }) as BoxOp)
}
