use anyhow::bail;
use onnx::onnx_pb;
use std::borrow::Cow;

use crate::{
    as_std, pvec, validate_providers, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, PVec,
    RealizedOp, Tensor,
};
#[derive(Debug, Clone)]
pub struct Unsqueeze {
    pub axes: Option<Vec<i64>>,
}

impl Unsqueeze {
    pub fn unsqueeze<D: DataType + ndarray::LinalgScalar + std::cmp::PartialOrd + num::NumCast>(
        &self,
        input: &Tensor,
        mut axes: Vec<i64>,
    ) -> anyhow::Result<Tensor> {
        axes.sort_by(|a, b| a.partial_cmp(b).expect("Failed to sort."));
        let mut new_shape = input.shape.clone();

        axes.iter().for_each(|new_axis| {
            new_shape.insert(num::cast(*new_axis).unwrap(), 1);
        });

        Ok(Tensor::new(input.dt, new_shape))
    }
}

impl Op for Unsqueeze {
    fn name(&self) -> Cow<str> {
        "Unsqueeze".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 1, 2, &self.name())?;
        if providers.len() == 1 && self.axes.is_none() {
            bail!("Invalid parameters for Unsqueeze.")
        }

        let axes = if let Some(ax) = &self.axes {
            ax.clone()
        } else {
            providers[1].as_slice()?.to_vec()
        };

        let new_tensor = as_std!(Unsqueeze::unsqueeze(providers[0].dt)(
            self,
            &providers[0],
            axes
        ))?;

        Ok(RealizedOp::zero_cost(pvec![new_tensor.into_arc_tensor()]))
    }
}

pub fn build_unsqueeze(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axes = proto.get_attribute("axes", None).ok();
    Ok(Box::new(Unsqueeze { axes }) as BoxOp)
}
