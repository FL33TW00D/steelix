use crate::ir::ops::shape::multi_broadcast;
use crate::prelude::*;
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct Sum;

impl Op for Sum {
    fn name(&self) -> Cow<str> {
        "Sum".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        let broadcasted_shape = multi_broadcast(
            &providers
                .iter()
                .map(|p| p.shape.clone())
                .collect::<Vec<_>>(),
        )
        .expect("Sum: broadcast failed");

        let res = Tensor::new(providers[0].dt, broadcasted_shape);

        Ok(RealizedOp {
            cost: OpCost {
                flops: providers.iter().fold(0, |acc, p| acc + p.numel()),
                ..OpCost::default()
            },
            outputs: pvec![res.into_arc_tensor()],
        })
    }
}
