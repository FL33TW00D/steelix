//Operator set is defined here: https://github.com/onnx/onnx/blob/main/onnx/defs/operator_sets.h
#![feature(get_mut_unchecked)]
mod model;
mod op_group;
mod op_node;
mod op_register;
mod tensor;
mod tensor_shape;
mod value_info;

pub mod ops;

use anyhow::bail;
use smallvec::{smallvec, SmallVec};
use std::{borrow::Cow, sync::Arc};

pub use model::*;
pub use op_group::*;
pub use op_node::*;
pub use op_register::*;
pub use tensor::*;
pub use tensor_shape::*;
pub use value_info::*;

#[derive(Debug, Default)]
pub struct OpCost {
    pub flops: usize,      //# Floating Point Operations
    pub parameters: usize, //# Parameters
}

impl OpCost {
    pub fn zero_cost() -> OpCost {
        OpCost::default()
    }

    pub fn unary_op_flops(input: &Tensor, flops_per_elem: usize) -> OpCost {
        OpCost {
            flops: input.numel() * flops_per_elem,
            parameters: 0,
        }
    }
}

type PVec = SmallVec<[Arc<Tensor>; 4]>;

type Shape = SmallVec<[usize; 4]>;

type StResult<T> = anyhow::Result<T>;

#[derive(Debug, Default)]
pub struct RealizedOp {
    cost: OpCost,
    outputs: PVec,
}

impl RealizedOp {
    pub fn zero_cost(outputs: PVec) -> RealizedOp {
        Self {
            cost: OpCost::default(),
            outputs,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum OpError {
    #[error("{0}")]
    ValidationError(String),
    #[error(transparent)]
    UnexpectedError(#[from] anyhow::Error),
}

pub trait Op {
    fn name(&self) -> Cow<str>;

    fn op_group(&self) -> OpGroup;

    ///Computes the cost of the operation and propagates the tensors forward
    ///with the appropriate shape updates
    fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp>;

    fn update(&mut self, _t: Arc<Tensor>) {}
}

pub type BoxOp = Box<dyn Op>;

pub fn validate_providers(
    providers: &PVec,
    lower: usize,
    upper: usize,
    name: &str,
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

elementwise!(Abs, Logic, 1);
elementwise!(Erf, Logic, 2);
elementwise!(Sigmoid, Logic, 4);
elementwise!(LeakyRelu, Activation, 2);
elementwise!(Not, Logic, 1);

#[macro_export]
macro_rules! elementwise {
    ($Op:ident, $group:ident, $flop:literal) => {
        #[derive(Debug, Clone)]
        pub struct $Op;

        impl $crate::Op for $Op {
            fn name(&self) -> Cow<str> {
                stringify!($Op).into()
            }

            fn op_group(&self) -> OpGroup {
                OpGroup::$group
            }

            fn realize(&self, providers: PVec) -> StResult<RealizedOp> {
                validate_providers(&providers, 1, 1, stringify!($Op))?;
                Ok(RealizedOp {
                    cost: OpCost::unary_op_flops(&providers[0], $flop),
                    outputs: smallvec![providers[0].clone()],
                })
            }
        }
    };
}
