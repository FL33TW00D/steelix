//Operator set is defined here: https://github.com/onnx/onnx/blob/main/onnx/defs/operator_sets.h
mod model;
mod op_group;
mod op_node;
mod op_register;
mod shape;
mod tensor;
mod value_info;

pub mod ops;

use anyhow::bail;
use smallvec::SmallVec;
use std::{borrow::Cow, sync::Arc};

pub use model::*;
pub use op_group::*;
pub use op_node::*;
pub use op_register::*;
pub use shape::*;
pub use tensor::*;
pub use value_info::*;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct OpCost {
    pub flops: usize,
    pub parameters: usize,
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

pub type PVec = SmallVec<[Arc<Tensor>; 4]>;

#[derive(Debug, Default)]
pub struct RealizedOp {
    pub cost: OpCost,
    pub outputs: PVec,
}

impl RealizedOp {
    pub fn zero_cost(outputs: PVec) -> RealizedOp {
        Self {
            cost: OpCost::default(),
            outputs,
        }
    }
}

impl PartialEq for RealizedOp {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.outputs == other.outputs
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

#[macro_export]
macro_rules! elementwise {
    ($Op:ident, $group:ident, $flop:literal) => {
        #[derive(Debug, Clone)]
        pub struct $Op;

        impl $crate::ir::Op for $Op {
            fn name(&self) -> Cow<str> {
                stringify!($Op).into()
            }

            fn op_group(&self) -> OpGroup {
                OpGroup::$group
            }

            fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
                validate_providers(&providers, 1, 1, stringify!($Op))?;
                Ok(RealizedOp {
                    cost: OpCost::unary_op_flops(&providers[0], $flop),
                    outputs: pvec![providers[0].clone()],
                })
            }
        }
    };
}

#[macro_export]
macro_rules! shape {
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::Shape::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ shape!(@one $x))*;
        #[allow(unused_mut)]
        let mut vec = smallvec::SmallVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            $crate::Shape(vec)
        } else {
            $crate::Shape(smallvec::SmallVec::from_vec(vec![$($x,)*]))
        }
    });
}

#[macro_export]
macro_rules! pvec {
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::PVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ pvec!(@one $x))*;
        #[allow(unused_mut)]
        let mut vec = $crate::PVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::PVec::from_vec(vec![$($x,)*])
        }
    });
}

//Elements that do not transform the shape, and purely cost compute
elementwise!(Abs, Logic, 1);
elementwise!(Erf, Logic, 2);
elementwise!(Sigmoid, Logic, 4);
elementwise!(LeakyRelu, Activation, 2);
elementwise!(Relu, Activation, 1);
elementwise!(Not, Logic, 1);
elementwise!(Elu, Activation, 1);
