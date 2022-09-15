//Operator set is defined here: https://github.com/onnx/onnx/blob/main/onnx/defs/operator_sets.h
mod helpers;
mod model;
mod op_group;
mod op_node;
mod op_register;
mod tensor;
mod tensor_shape;
mod value_info;

pub mod ops;

use anyhow::bail;
use smallvec::SmallVec;
use std::{borrow::Cow, sync::Arc};

pub use helpers::*;
pub use model::*;
pub use op_group::*;
pub use op_node::*;
pub use op_register::*;
pub use tensor::*;
pub use tensor_shape::*;
pub use value_info::*;

#[derive(Debug, Default)]
pub struct OpCost {
    pub mac: usize,        //# Multiply Accumulate Ops
    pub parameters: usize, //# Parameters
    pub flops: usize,      //# Floating Point Operations
}

type QuadVec = SmallVec<[Arc<Tensor>; 4]>;

#[derive(Debug, Default)]
pub struct RealizedOp {
    cost: OpCost,
    outputs: QuadVec,
}

pub trait Op {
    fn name(&self) -> Cow<str>;

    fn op_group(&self) -> OpGroup;

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp>;

    fn update(&mut self, _t: Arc<Tensor>) {}
}

pub type BoxOp = Box<dyn Op>;

pub fn validate_providers(
    providers: &QuadVec,
    lower: usize,
    upper: usize,
    name: String,
) -> anyhow::Result<()> {
    if providers.len() > upper || providers.len() < lower {
        bail!(
            "Expected between {} and {} providers, got: {} in operation: {}",
            lower,
            upper,
            providers.len(),
            name
        )
    } else {
        Ok(())
    }
}
