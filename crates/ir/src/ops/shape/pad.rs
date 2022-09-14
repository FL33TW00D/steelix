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

impl Pad {}
