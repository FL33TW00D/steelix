use bytes::BufMut;
use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{
    validate_providers, BoxOp, DType, IntoArcTensor, Op, OpGroup, PVec, RealizedOp, Tensor,
};
#[derive(Debug, Clone)]
pub struct Shape {
    start: i64,
    end: i64,
}

impl Op for Shape {
    fn name(&self) -> Cow<str> {
        "Shape".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 1, &self.name())?;
        let input_shape = &providers[0].shape;
        let end = if self.end == -1 {
            input_shape.len() as i64
        } else {
            self.end
        };
        let new_shape = providers[0].shape[self.start as usize..end as usize]
            .iter()
            .cloned()
            .map(|i| i as i64)
            .collect::<Vec<i64>>();

        let out = Tensor::from_vec(smallvec![new_shape.len()], new_shape);
        println!("SHAPE OUTPUT: {:?}", out);
        Ok(RealizedOp::zero_cost(smallvec![out.into_arc_tensor()]))
    }
}

pub fn build_shape(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let start = proto.get_attribute("start", Some(0))? as i64;
    let end = proto.get_attribute("end", Some(-1))? as i64;
    Ok(Box::new(Shape { start, end }) as BoxOp)
}
