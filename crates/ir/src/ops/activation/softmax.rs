use std::borrow::Cow;

use onnx::onnx_pb;
use smallvec::smallvec;

use crate::{BoxOp, IntoArcTensor, Op, OpCost, OpGroup, QuadVec, RealizedOp, Tensor};

#[derive(Debug, Clone)]
pub struct Softmax {
    pub axis: i64,
}

/*
def calculate_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])
*/

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Activation
    }

    // Approximate flops breakdown:
    //   2*n          -- compute shifted logits
    //   n            -- exp of shifted logits
    //   2*n          -- compute softmax from exp of shifted logits
    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        let output_shape = if self.axis == -1 {
            providers[0].shape[providers[0].shape.len()]
        } else {
            providers[0].shape[self.axis as usize]
        };
        let out = Tensor::zeros::<f32>(vec![output_shape]);
        Ok(RealizedOp {
            cost: OpCost {
                mac: 0,
                parameters: 0,
            },
            outputs: smallvec![out.into_arc_tensor(); 4],
        })
    }
}

pub fn build_softmax(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.extract_named_int("axis")?.unwrap_or(-1);
    Ok(Box::new(Softmax { axis }) as BoxOp)
}
