use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, RealizedOp, Tensor};

use smallvec::smallvec;

#[derive(Debug, Clone)]
pub struct GlobalAveragePool;

impl Op for GlobalAveragePool {
    fn name(&self) -> Cow<str> {
        "GlobalAveragePool".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Pool
    }

    fn realize(&self, providers: crate::PVec) -> anyhow::Result<crate::RealizedOp> {
        let input_shape = &providers[0].shape;
        let out_shape = vec![input_shape[0], input_shape[1], 1, 1];
        let out = Tensor::new(providers[0].dt, out_shape.into());
        Ok(RealizedOp {
            cost: OpCost {
                flops: providers[0].numel(),
                parameters: 0,
            },
            outputs: smallvec![out.into_arc_tensor()],
        })
    }
}

pub fn build_globalavgpool(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(GlobalAveragePool) as BoxOp)
}
