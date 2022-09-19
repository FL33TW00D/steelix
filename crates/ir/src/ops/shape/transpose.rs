use onnx::onnx_pb;
use smallvec::smallvec;
use std::borrow::Cow;

use crate::{
    validate_providers, BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp, Tensor,
};

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

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 1, &self.name())?;

        let new_shape = Self::transpose::<f32>(self, &providers[0], &self.perm);
        Ok(RealizedOp::zero_cost(smallvec![Tensor::new(
            crate::DType::F32,
            new_shape.into()
        )
        .into_arc_tensor()]))
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
