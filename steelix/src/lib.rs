#![feature(vec_into_raw_parts)]
mod build_cli;
mod display;
mod ir;
mod parser;

pub use build_cli::*;
pub use display::*;
pub use ir::*;
pub use parser::*;

mod prelude {
    pub use crate::ir::{
        validate_providers, BoxOp, DType, DataType, IntoArcTensor, Op, OpCost, OpGroup, PVec,
        RealizedOp, Shape, Tensor,
    };
    pub use crate::{as_std, pvec, shape};
}
