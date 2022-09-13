use std::{borrow::Cow, sync::Arc};

use ndarray::Array;
use onnx::onnx_pb;

use crate::{as_std, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Transpose {
    pairs: Vec<(usize, usize)>,
    perm: Vec<usize>,
}

impl Transpose {
    fn build_pairs(perm: Vec<usize>) -> Vec<(usize, usize)> {
        let mut pairs = vec![];
        for (from, to) in (0..4).zip(perm) {
            if from != to {
                pairs.push((from, to));
            }
        }
        pairs
    }
    pub fn transpose<D: DataType + ndarray::LinalgScalar>(
        &self,
        input: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>> {
        let mut tt = input.to_owned();
        let a_t = tt
            .to_array_view_mut::<D>()
            .unwrap()
            .permuted_axes(self.perm.clone());
        let mut trans = Array::zeros(a_t.raw_dim());
        trans.assign(&a_t);
        Ok(trans.into_arc_tensor())
    }
}

impl Op for Transpose {
    fn name(&self) -> Cow<str> {
        "Transpose".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        Ok(vec![as_std!(Transpose::transpose(providers[0].dt)(
            self,
            &providers[0]
        ))?])
    }
}

pub fn build_transpose(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let perm: Vec<usize> = proto
        .extract_named_intv("perm")?
        .unwrap()
        .iter()
        .map(|&e| e as usize)
        .collect();
    let pairs = Transpose::build_pairs(perm.clone());
    Ok(Box::new(Transpose { pairs, perm }) as BoxOp)
}
