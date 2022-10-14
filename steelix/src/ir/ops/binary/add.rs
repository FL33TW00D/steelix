use std::borrow::Cow;

use crate::{ir::ops::shape::multi_broadcast, prelude::*};
use onnx::onnx_pb;

#[derive(Debug, Clone)]
pub struct Add;

impl Op for Add {
    fn name(&self) -> Cow<str> {
        "Add".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Tensor
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 2, &self.name())?;
        let broadcasted_shape = multi_broadcast(
            &providers
                .iter()
                .map(|p| p.shape.clone())
                .collect::<Vec<_>>(),
        )
        .expect("Failed to broadcast shapes in Add");

        let res = Tensor::new(providers[0].dt, broadcasted_shape);
        Ok(RealizedOp {
            cost: OpCost {
                flops: providers[0].numel(),
                ..OpCost::default()
            },
            outputs: pvec![res.into_arc_tensor()],
        })
    }
}

pub fn build_add(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Add) as BoxOp)
}
