

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

impl<O: Op> OpNode<O> {
    #[inline]
    pub fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        self.op.realize(providers)
    }
}
