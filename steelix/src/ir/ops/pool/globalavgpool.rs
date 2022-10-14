use std::borrow::Cow;

use crate::prelude::*;
use onnx::onnx_pb;

#[derive(Debug, Clone)]
pub struct GlobalAveragePool;

impl Op for GlobalAveragePool {
    fn name(&self) -> Cow<str> {
        "GlobalAveragePool".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Pool
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        let input_shape = &providers[0].shape;
        let out = Tensor::new(
            providers[0].dt,
            shape![input_shape[0], input_shape[1], 1, 1],
        );
        Ok(RealizedOp {
            cost: OpCost {
                flops: providers[0].numel(),
                ..OpCost::default()
            },
            outputs: pvec![out.into_arc_tensor()],
        })
    }
}

pub fn build_globalavgpool(_: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(GlobalAveragePool) as BoxOp)
}
