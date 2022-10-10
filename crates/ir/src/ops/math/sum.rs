use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

use crate::{
    ops::shape::multi_broadcast, BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp,
    Tensor,
};

#[derive(Debug, Clone)]
pub struct Sum;

impl Op for Sum {
    fn name(&self) -> Cow<str> {
        "Sum".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<crate::RealizedOp> {
        let broadcasted_shape = multi_broadcast(
            &providers
                .iter()
                .map(|p| p.shape.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let res = Tensor::new(providers[0].dt, broadcasted_shape);

        Ok(RealizedOp {
            cost: OpCost {
                flops: 0, //TODO fix
                parameters: 0,
            },
            outputs: smallvec![res.into_arc_tensor()],
        })
    }
}

pub fn build_sum(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Sum) as BoxOp)
}
