use std::borrow::Cow;

use crate::ir::ops::shape::multi_broadcast;
use crate::prelude::*;
use anyhow::bail;
use onnx::onnx_pb;
use smallvec::smallvec;

#[derive(Debug, Clone)]
pub struct Gemm {
    #[allow(dead_code)]
    trans_a: usize,
    #[allow(dead_code)]
    trans_b: usize,
}

impl Gemm {
    fn compute_cost(
        &self,
        a_shape: Shape,
        b_shape: Shape,
        ab_shape: Shape,
        c_shape: Shape,
    ) -> OpCost {
        let m = a_shape[0];
        let n = a_shape[1];
        let p = b_shape[1];
        let ab_flops = m * n * (2 * p - 1);
        let ab_c_flops = ab_shape[0] * ab_shape[1] * (2 * c_shape[1] - 1);
        OpCost {
            flops: ab_flops + ab_c_flops,
            ..Default::default()
        }
    }
}

impl Op for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    //TODO: support transpose
    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 3, &self.name())?;

        let a = &providers[0];
        let b = &providers[1];
        let c = if providers.len() == 2 {
            Tensor::new(a.dt, shape!(1)).into_arc_tensor()
        } else {
            providers[2].clone()
        };

        let a_shape = &a.shape;
        let b_shape = &b.shape;

        let matching_dim = |a_shape: &Shape, b_shape: &Shape| -> anyhow::Result<usize> {
            for i in 0..a_shape.len() {
                if a_shape[i] == b_shape[i] {
                    return Ok(i);
                }
            }
            bail!(
                "GEMM: No equal dimension found in {:?} and {:?}",
                a_shape,
                b_shape
            );
        };

        let ab_shape = {
            let mut ab_shape = smallvec![0;2];
            let mut a_shape = a_shape.clone();
            let mut b_shape = b_shape.clone();
            let a_dim = matching_dim(&a_shape, &b_shape)?;
            let b_dim = matching_dim(&b_shape, &a_shape)?;
            a_shape.remove(a_dim);
            b_shape.remove(b_dim);
            ab_shape[0] = a_shape[0];
            ab_shape[1] = b_shape[0];
            Shape(ab_shape)
        };

        let c_shape = multi_broadcast(&[ab_shape.clone(), c.shape.clone()])
            .expect("Could not broadcast C -> A*B in GEMM");

        let res = Tensor::new(providers[0].dt, ab_shape.clone());

        Ok(RealizedOp {
            cost: self.compute_cost(a_shape.clone(), b_shape.clone(), ab_shape, c_shape),
            outputs: smallvec![res.into_arc_tensor()],
        })
    }
}

pub fn build_gemm(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let trans_a = proto.get_attribute("transA", Some(0))? as usize;
    let trans_b = proto.get_attribute("transB", Some(0))? as usize;
    Ok(Box::new(Gemm { trans_a, trans_b }) as BoxOp)
}
