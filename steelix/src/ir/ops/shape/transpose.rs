use steelix_onnx::onnx_pb;
use std::borrow::Cow;

use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct Transpose {
    perm: Vec<usize>,
}

impl Transpose {
    fn transpose(&self, input: &Tensor, axes: &[usize]) -> Vec<usize> {
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

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 1, &self.name())?;

        let transposed_shape = self.transpose(&providers[0], &self.perm).into();
        let result = Tensor::new(providers[0].dt, Shape(transposed_shape)).into_arc_tensor();

        Ok(RealizedOp::zero_cost(pvec!(result)))
    }
}

pub fn build_transpose(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let perm: Vec<usize> = proto
        .get_attribute::<Vec<i64>>("perm", None)?
        .iter()
        .cloned()
        .map(|x| x as usize)
        .collect();
    Ok(Box::new(Transpose { perm }) as BoxOp)
}
