use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, RealizedOp, Tensor};

#[derive(Debug, Clone)]
pub struct Squeeze {
    pub axes: Option<Vec<usize>>,
}

impl Squeeze {
    pub fn squeeze(&self, to_squeeze: &Tensor) -> Vec<usize> {
        let shape_iter = to_squeeze.shape.iter();
        let new_shape: Vec<usize> = if self.axes.is_some() {
            let all_axes = self.axes.as_ref().unwrap();
            shape_iter
                .enumerate()
                .filter(|(idx, _)| !all_axes.contains(idx))
                .map(|tup| to_squeeze.shape[tup.0])
                .collect()
        } else {
            shape_iter.filter(|ax| **ax != 1_usize).copied().collect()
        };

        new_shape
    }
}

impl Op for Squeeze {
    fn name(&self) -> Cow<str> {
        "Squeeze".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn cost(&self, providers: crate::QuadVec) -> anyhow::Result<crate::RealizedOp> {
        let new_shape = self.squeeze(&providers[0]);
        let output = Tensor::zeros::<f32>(new_shape);
        Ok(RealizedOp {
            cost: OpCost::zero_cost(),
            outputs: smallvec![output.into_arc_tensor(); 4],
        })
    }
}

pub fn build_squeeze(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axes = proto.extract_named_attr("axes")?.unwrap();
    Ok(Box::new(Squeeze {
        axes: Some(axes.ints.clone().iter().map(|&i| i as usize).collect()),
    }) as BoxOp)
}
