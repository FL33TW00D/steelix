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
    pub mac: usize,        //# Multiply Accumulate Ops
    pub parameters: usize, //# Parameters
}

impl OpCost {
    pub fn zero_cost() -> OpCost {
        OpCost::default()
    }
}

type QuadVec = SmallVec<[Arc<Tensor>; 4]>;

type Shape = SmallVec<[usize; 4]>;

type StResult<T> = anyhow::Result<T>;

#[derive(Debug, Default)]
pub struct RealizedOp {
    cost: OpCost,
    outputs: QuadVec,
}

impl RealizedOp {
    pub fn zero_cost(outputs: QuadVec) -> RealizedOp {
        Self {
            cost: OpCost::default(), //usize defaults to 0
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
    fn realize(&self, providers: QuadVec) -> anyhow::Result<RealizedOp>;

    fn update(&mut self, _t: Arc<Tensor>) {}

    fn param_count(&self, providers: QuadVec) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn mac_count(&self, providers: QuadVec) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn output_shape(&self, providers: QuadVec) -> anyhow::Result<Shape> {
        Ok(smallvec![0, 0, 0, 0])
    }
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

//What do we want the macro to do? To implement our Op trait for us. We will still need to define
//structs for everyone because they have different fields.

pub struct Abs;

#[macro_export]
macro_rules! elementwise {
    ($name:ident, $group:ident, $( [$($typ:ident),*] => $cab:expr),*) => {
        impl $crate::Op for $name {
            fn name(&self) -> Cow<str> {
                $name.into()
            }

            fn op_group(&self) -> OpGroup {
                OpGroup::$group
            }

            fn realize(&self, providers: QuadVec) -> StResult {
                validate_providers(&providers, 1, 1, $name)?;

                //validate providers
                //calculate costs
                //calculate output shape
                //create output tensor
            }
        }
    };
}
