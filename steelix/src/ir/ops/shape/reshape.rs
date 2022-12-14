use crate::prelude::*;
use std::borrow::Cow;
use steelix_onnx::onnx_pb;

#[derive(Debug, Clone)]
pub struct Reshape {
    pub allow_zero: i64,
}

//Reshape is a problematic operation, as it depends on the data contents of the tensor
impl Reshape {
    pub fn reshape<D: DataType + ndarray::LinalgScalar + num::NumCast>(
        original_shape: Shape,
        shape_tensor: &Tensor,
    ) -> Shape {
        let mut shape_data: Vec<D> = shape_tensor.as_slice().unwrap().into();

        let mut product = D::one();
        let mut unknown_dim = None;
        for (i, dim) in shape_data.iter().enumerate() {
            if *dim == D::from(-1).unwrap() {
                if unknown_dim.is_some() {
                    panic!("Reshape: only one unknown dimension is allowed");
                }
                unknown_dim = Some(i);
            } else if *dim != D::zero() {
                product = product * *dim;
            }
        }

        if let Some(unknown_dim) = unknown_dim {
            shape_data[unknown_dim] =
                D::from(original_shape.iter().product::<usize>() / product.to_usize().unwrap())
                    .expect("Failed to compute unknown dimension in Reshape");
        }

        let mut new_shape = shape!();
        for elem in shape_data {
            new_shape.push(num::cast(elem).unwrap_or_else(|| panic!("Failed to cast: {:?}", elem)));
        }

        new_shape
    }
}

impl Op for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        validate_providers(&providers, 2, 2, &self.name())?;
        let new_shape = as_std!(Reshape::reshape(providers[1].dt)(
            providers[0].shape.clone(),
            &providers[1]
        ));

        let reshaped = Tensor::new(providers[0].dt, new_shape).into_arc_tensor();

        Ok(RealizedOp::zero_cost(pvec![reshaped]))
    }
}

pub fn build_reshape(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let allow_zero = proto.get_attribute("allowzero", Some(0))?;
    Ok(Box::new(Reshape { allow_zero }) as BoxOp)
}
