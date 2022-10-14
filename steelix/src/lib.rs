#![feature(vec_into_raw_parts)]
mod display;
mod ir;
mod parser;

mod prelude {
    pub use crate::ir::{
        validate_providers, BoxOp, DType, DataType, IntoArcTensor, Op, OpCost, OpGroup, PVec,
        RealizedOp, Shape, Tensor,
    };
    pub use crate::{as_std, pvec, shape};
}
