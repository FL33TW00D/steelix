use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

use crate::{
    ops::shape::multi_broadcast, validate_providers, BoxOp, IntoArcTensor, Op, OpCost, OpGroup,
    PVec, RealizedOp, Tensor,
};

#[derive(Debug, Clone)]
pub struct Gemm {
    trans_a: usize,
    trans_b: usize,
}

impl Op for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<crate::RealizedOp> {
        validate_providers(&providers, 2, 3, &self.name())?;
        let broadcasted_shape = multi_broadcast(
            &providers
                .iter()
                .map(|p| p.shape.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let p0_shape = &broadcasted_shape;
        let p1_shape = &broadcasted_shape;

        let m = p0_shape[0];
        let n = p1_shape[1];
        let p = p0_shape[1];

        let res = Tensor::new(providers[0].dt, broadcasted_shape);

        Ok(RealizedOp {
            cost: OpCost {
                flops: m * n * (2 * p - 1),
                parameters: 0,
            },
            outputs: smallvec![res.into_arc_tensor()],
        })
    }
}

pub fn build_gemm(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let trans_a = proto.get_attribute("transA", Some(0))? as usize;
    let trans_b = proto.get_attribute("transB", Some(0))? as usize;
    Ok(Box::new(Gemm { trans_a, trans_b }) as BoxOp)
}
