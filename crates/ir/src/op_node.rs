use crate::{Op, Tensor};
use std::sync::Arc;

#[derive(Debug, Clone, Default, Hash)]
pub struct OpNode<O: Op> {
    pub id: usize,
    pub name: String,
    pub providers: Vec<usize>,
    pub consumers: Vec<usize>,
    pub op: O,
}

impl<O: Op> OpNode<O> {}
