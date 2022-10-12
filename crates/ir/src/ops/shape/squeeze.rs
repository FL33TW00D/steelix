use std::borrow::Cow;

use onnx::onnx_pb;

use crate::{pvec, BoxOp, IntoArcTensor, Op, OpCost, OpGroup, PVec, RealizedOp, Shape, Tensor};

#[derive(Debug, Clone)]
pub struct Squeeze {
    pub axes: Option<Vec<usize>>,
}

impl Squeeze {
    pub fn squeeze(&self, to_squeeze: &Tensor) -> Shape {
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

        Shape(new_shape.into())
    }
}

impl Op for Squeeze {
    fn name(&self) -> Cow<str> {
        "Squeeze".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        let new_shape = self.squeeze(&providers[0]);
        let output = Tensor::new(providers[0].dt, new_shape);
        Ok(RealizedOp {
            cost: OpCost::zero_cost(),
            outputs: pvec![output.into_arc_tensor()],
        })
    }
}

pub fn build_squeeze(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axes: Vec<i64> = proto.get_attribute("axes", None)?;
    Ok(Box::new(Squeeze {
        axes: Some(axes.iter().cloned().map(|i| i as usize).collect()),
    }) as BoxOp)
}
