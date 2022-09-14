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

#[derive(Debug)]
pub struct OpCost {
    pub mac: usize,        //# Multiply Accumulate Ops
    pub parameters: usize, //# Parameters
    pub flops: usize,      //# Floating Point Operations
}

type QuadVec = SmallVec<[Arc<Tensor>; 4]>;

#[derive(Debug)]
pub struct RealizedOp {
    cost: OpCost,
    outputs: QuadVec,
}

pub trait Op {
    fn name(&self) -> Cow<str>;

    fn op_group(&self) -> OpGroup;

    fn cost(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        Ok(RealizedOp {
            cost: OpCost {
                mac: 0,
                parameters: 0,
                flops: 0,
            },
            outputs: QuadVec::new(),
        })
    }
}

pub type BoxOp = Box<dyn Op>;

#[macro_export]
macro_rules! provider_bounds {
    // `()` indicates that the macro takes no argument.
    ($providers:ident, $lower:expr, $upper:expr, $op:ident) => {
        // The macro will expand into the contents of this block.
        if $providers.len() > $upper || $providers.len() < $lower {
            bail!(
                "Expected between {} and {} providers, got: {} in operation: {}",
                $lower,
                $upper,
                $providers.len(),
                $op.name()
            )
        }
    };
}
