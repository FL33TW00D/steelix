use ndarray::{ArrayD, IxDyn, SliceInfo, SliceInfoElem};
use strum_macros::EnumString;

use crate::{DataType, IntoTensor, Tensor};

#[derive(EnumString)]
pub enum PaddingMode {
    Constant,
    Reflect,
    Edge,
}

#[derive(Debug, Clone)]
pub struct Pad {
    pads: Vec<usize>,
}

impl Pad {
    fn convert_pads(pads: Vec<i64>) -> Vec<(usize, usize)> {
        let midpoint = pads.len() / 2;
        pads[..midpoint]
            .iter()
            .map(|&e| e as usize)
            .zip(pads[midpoint..].iter().map(|&e| e as usize))
            .collect()
    }

    pub fn pad<T: Copy + DataType + ndarray::LinalgScalar>(
        input_tensor: &Tensor,
        pads: Vec<i64>,
    ) -> anyhow::Result<Tensor> {
        let mut converted = Self::convert_pads(pads);
        //here, we need to push 0 padding for batch, channel
        for _ in 0..2 {
            converted.insert(0, (0, 0));
        }
        let input = input_tensor.to_array_view::<T>()?;
        let output_shape: Vec<usize> = input
            .shape()
            .iter()
            .zip(converted.iter())
            .map(|(&d, &(a, b))| d + a + b)
            .collect();

        let mut output = ArrayD::<T>::from_elem(output_shape, T::zero());

        let slice_spec: Vec<SliceInfoElem> = converted
            .iter()
            .map(|&(a, b)| SliceInfoElem::Slice {
                start: a as isize,
                end: if b != 0 { Some(-(b as isize)) } else { None },
                step: 1,
            })
            .collect();
        let slice_info = SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_spec).unwrap();
        output.slice_mut(slice_info.as_ref()).assign(&input);
        Ok(output.into_tensor())
    }
}
