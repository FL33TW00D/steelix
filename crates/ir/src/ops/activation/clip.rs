use std::{borrow::Cow, sync::Arc};

use onnx::onnx_pb;

use crate::{as_float, BoxOp, DType, DataType, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct Clip {
    pub min: Option<i64>,
    pub max: Option<i64>,
}

impl Clip {
    #[inline]
    fn clamp<T: DataType + Copy + ndarray::LinalgScalar + std::cmp::PartialOrd>(
        mut x: T,
        min: T,
        max: T,
    ) -> T {
        if x < min {
            x = min;
        }
        if x > max {
            x = max;
        }
        x
    }

    //Partial ord bit dangerous here?
    //USE FLOAT CMP
    pub fn clip<T: DataType + Copy + ndarray::LinalgScalar + std::cmp::PartialOrd>(
        &self,
        input: &mut Tensor,
        min: &Tensor,
        max: &Tensor,
    ) {
        let iptr = input.as_mut_ptr::<T>().unwrap();

        let min_val = min.as_ptr::<T>().unwrap();
        let max_val = max.as_ptr::<T>().unwrap();
        for idx in 0..input.len {
            unsafe {
                *iptr.add(idx) = Self::clamp(*iptr.add(idx), *min_val, *max_val);
            }
        }
    }
}

impl Op for Clip {
    fn name(&self) -> Cow<str> {
        "Clip".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Activation
    }

    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        if providers.len() == 3 {
            unsafe {
                as_float!(Clip::clip(providers[0].dt)(
                    self,
                    Arc::get_mut_unchecked(&mut providers[0].clone()),
                    &providers[1],
                    &providers[2]
                ));
                Ok(vec![providers[0].clone()])
            }
        } else {
            anyhow::bail!("Pad had incorrect inputs.")
        }
    }
}

pub fn build_clip(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let _min = proto.extract_named_attr("min")?;
    let _max = proto.extract_named_attr("max")?;
    Ok(Box::new(Clip {
        min: Some(0),
        max: Some(0),
    }) as BoxOp)
}
