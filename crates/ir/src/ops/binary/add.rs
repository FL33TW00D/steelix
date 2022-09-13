use std::{borrow::Cow, ops::AddAssign, sync::Arc};

use onnx::onnx_pb;

use crate::{as_std, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Add;

impl Add {
    pub fn add<T: DataType + ndarray::LinalgScalar + AddAssign>(
        a: &Tensor,
        b: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>> {
        //So many owneds
        let mut a = a.clone();
        let mut a = a.to_array_view_mut::<T>().unwrap();

        a += &b.to_array_view().unwrap();
        Ok(a.to_owned().into_arc_tensor())
    }
}

impl Op for Add {
    fn name(&self) -> Cow<str> {
        "Add".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Tensor
    }

    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        Ok(vec![as_std!(Add::add(providers[0].dt)(
            &providers[0],
            &providers[1]
        ))?])
    }
}

pub fn build_add(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Add) as BoxOp)
}
