use std::{borrow::Cow, sync::Arc};

use ndarray::{Axis, Ix2};
use onnx::onnx_pb;

use crate::{as_float, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Softmax {
    pub axis: i64,
}

impl Softmax {
    fn max<T: PartialOrd>(a: T, b: T) -> T {
        if b > a {
            b
        } else {
            a
        }
    }
    pub fn softmax<
        T: DataType
            + std::cmp::PartialOrd
            + Copy
            + ndarray::LinalgScalar
            + num_traits::real::Real
            + num_traits::NumAssign,
    >(
        &self,
        input: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>> {
        //so many clones
        let mut clon = input.clone();
        let mut ind = clon
            .to_array_view_mut::<T>()
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let max = ind.fold_axis(Axis(self.axis as usize), T::min_value(), |&a, &b| {
            Self::max(a, b)
        });
        ind.indexed_iter_mut().for_each(|((idx, _), x)| {
            *x = (*x - max[idx]).exp();
        });
        let sum = ind.sum_axis(Axis(self.axis as usize));
        ind.indexed_iter_mut().for_each(|((idx, _), x)| {
            *x /= sum[idx];
        });
        Ok(ind.to_owned().into_arc_tensor())
    }
}

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Activation
    }

    ///Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    ///e_x = np.exp(x - np.max(x))
    ///return e_x / e_x.sum(axis=0)
    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        Ok(vec![as_float!(Softmax::softmax(providers[0].dt)(
            self,
            &providers[0]
        ))?])
    }
}

pub fn build_softmax(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.extract_named_int("axis")?.unwrap_or(-1);
    Ok(Box::new(Softmax { axis }) as BoxOp)
}
