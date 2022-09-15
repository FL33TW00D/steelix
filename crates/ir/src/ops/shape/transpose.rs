use onnx::onnx_pb;
use std::borrow::Cow;

use crate::{BoxOp, Op, OpGroup, QuadVec, RealizedOp};

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
}

impl Op for Transpose {
    fn name(&self) -> Cow<str> {
        "Transpose".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        //Here we need to return something real, and we need to know how the shape changed
        Ok(RealizedOp::default())
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
