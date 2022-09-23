use ndarray::{Axis, Dimension};
use onnx::onnx_pb;
use smallvec::smallvec;
use std::{borrow::Cow, sync::Arc};

use crate::{
    as_std, BoxOp, DType, DataType, IntoArcTensor, IntoTensor, Op, OpGroup, PVec, RealizedOp,
    Shape, Tensor,
};
#[derive(Debug, Clone)]
pub struct Gather {
    pub axis: i64,
}

impl Gather {
    pub fn compute_output_shape<D: num::NumCast + std::clone::Clone>(
        &self,
        input_shape: &[D],
        indices_shape: &[D],
    ) -> anyhow::Result<Shape> {
        let mut output_shape = smallvec![];
        for (idx, dim) in input_shape.iter().enumerate() {
            if idx as i64 != self.axis {
                output_shape.push(num::cast((*dim).clone()).unwrap());
            } else {
                for idx2 in indices_shape {
                    output_shape.push(num::cast((*idx2).clone()).unwrap());
                }
            }
        }
        Ok(output_shape)
    }

    unsafe fn eval<T: DataType + num_traits::Zero + num_traits::NumCast>(
        &self,
        data: Arc<Tensor>,
        indices: &Arc<Tensor>,
    ) -> anyhow::Result<Arc<Tensor>> {
        let data_view = data.to_array_view_unchecked::<T>();
        if indices.shape.len() == 0 {
            let mut index = *indices.to_scalar::<i64>()?;
            if index < 0 {
                index += data_view.shape()[0] as i64;
            }
            let mut tensor = data_view
                .index_axis(Axis(self.axis as usize), index as usize)
                .to_owned()
                .into_tensor();
            return Ok(tensor.into_arc_tensor());
        }

        let mut output =
            Tensor::uninitialized::<T>(self.compute_output_shape(&data.shape, &indices.shape)?);

        println!("OUTPUT SHAPE: {:?}", output);
        let mut view = output.to_array_view_mut_unchecked::<T>();
        for (indices_coords, indices_value) in indices.to_array_view::<i64>()?.indexed_iter() {
            println!("INDEX COORDS: {:?}", indices_coords);
            println!("INDICES VALUE: {:?}", indices_value);
            let mut to_update = view.index_axis_mut(Axis(self.axis as usize), indices_coords[0]);
            for idx in 1..indices_coords.ndim() {
                to_update = to_update.index_axis_move(Axis(0), indices_coords[idx]);
            }
            let index_value = if *indices_value >= 0 {
                *indices_value
            } else {
                indices_value + data_view.shape()[self.axis as usize] as i64
            } as usize;
            to_update.assign(&data_view.index_axis(Axis(self.axis as usize), index_value));
        }
        Ok(output.into_arc_tensor())
    }
}

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Shape
    }

    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        //print out all providers one by one

        unsafe {
            let result = as_std!(Self::eval(providers[0].dt)(
                self,
                providers[0].clone(),
                &providers[1]
            ))?;
            Ok(RealizedOp::zero_cost(smallvec![result]))
        }
    }
}

pub fn build_gather(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let axis = proto.get_attribute("axis", Some(0))?;
    Ok(Box::new(Gather { axis }) as BoxOp)
}
