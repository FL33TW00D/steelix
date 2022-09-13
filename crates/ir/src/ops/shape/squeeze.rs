use std::{borrow::Cow, sync::Arc};

use onnx::onnx_pb;

use crate::{as_std, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Squeeze {
    pub axes: Option<Vec<usize>>,
}

impl Squeeze {
    pub fn squeeze<T>(&self, to_squeeze: &mut Tensor)
    where
        T: DataType,
    {
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

        to_squeeze.update_shape(new_shape);
    }
}

impl Op for Squeeze {
    fn name(&self) -> Cow<str> {
        "Squeeze".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, mut providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        unsafe {
            as_std!(Squeeze::squeeze(providers[0].dt)(
                self,
                Arc::get_mut_unchecked(&mut providers[0])
            ));
            Ok(vec![providers[0].clone()])
        }
    }
}

pub fn build_squeeze(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axes = proto.extract_named_attr("axes")?.unwrap();
    Ok(Box::new(Squeeze {
        axes: Some(axes.ints.clone().iter().map(|&i| i as usize).collect()),
    }) as BoxOp)
}
