//Operator set is defined here: https://github.com/onnx/onnx/blob/main/onnx/defs/operator_sets.h
#![feature(get_mut_unchecked)]
extern crate blas_src;
mod helpers;
mod model;
mod op_group;
mod op_node;
mod op_register;
mod tensor;
mod tensor_shape;
mod value_info;

pub mod ops;

use std::{borrow::Cow, sync::Arc};

pub use helpers::*;
pub use model::*;
pub use op_group::*;
pub use op_node::*;
pub use op_register::*;
pub use tensor::*;
pub use tensor_shape::*;
pub use value_info::*;

pub trait Op {
    fn name(&self) -> Cow<str>;

    fn op_group(&self) -> OpGroup;

    fn mutator(&self) -> bool {
        false
    }

    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>>;

    fn update(&mut self, _t: Arc<Tensor>) {}
}

pub type BoxOp = Box<dyn Op>;
