use onnx::onnx_pb;
use std::{borrow::Cow, sync::Arc};

use crate::{validate_providers, BoxOp, Op, OpCost, OpGroup, PVec, RealizedOp, Tensor};

#[derive(Debug, Clone)]
pub struct Transpose {
    perm: Vec<usize>,
}

impl Transpose {
    fn transpose<D>(&self, input: &Tensor, axes: &[usize]) -> Vec<usize> {
        let mut usage_counts = [0, 0, 0, 0];
        for axis in axes {
            usage_counts[*axis] += 1;
        }
        for count in usage_counts {
            assert_eq!(count, 1, "each axis must be listed exactly once");
        }
        let mut new_dim = usage_counts;
        {
            let dim = &input.shape;
            for (new_axis, &axis) in axes.iter().enumerate() {
                new_dim[new_axis] = dim[axis];
            }
        }
        new_dim.into()
    }
}

impl Op for Transpose {
    fn name(&self) -> Cow<str> {
        "Transpose".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, mut providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 1, &self.name())?;

        let new_shape = Self::transpose::<f32>(self, &providers[0], &self.perm);
        unsafe { Arc::get_mut_unchecked(&mut providers[0]).update_shape(new_shape.into()) };

        let mut result = PVec::new();
        result.push(providers[0].clone());
        Ok(RealizedOp {
            cost: OpCost {
                flops: 42,
                parameters: 0,
            },
            outputs: result,
        })
    }
}

pub fn build_transpose(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let perm: Vec<usize> = proto
        .get_attribute::<Vec<i64>>("perm", None, proto)?
        .iter()
        .cloned()
        .map(|x| x as usize)
        .collect();
    Ok(Box::new(Transpose { perm }) as BoxOp)
}
