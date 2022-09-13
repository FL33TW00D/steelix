use std::{borrow::Cow, sync::Arc};

use ndarray::Ix2;
use onnx::onnx_pb;

use crate::{as_std, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Matmul;

impl Matmul {
    pub fn matmul<T: DataType + ndarray::LinalgScalar>(
        a: &Tensor,
        b: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>> {
        let and = a
            .to_array_view::<T>()
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let bnd = b
            .to_array_view()
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();

        Ok(and.dot(&bnd).into_arc_tensor())
    }
}

impl Op for Matmul {
    fn name(&self) -> Cow<str> {
        "Matmul".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Transform
    }

    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        Ok(vec![as_std!(Matmul::matmul(providers[0].dt)(
            &providers[0],
            &providers[1]
        ))?])
    }
}

pub fn build_matmul(_proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Matmul) as BoxOp)
}
