use crate::{Op, PVec, RealizedOp};

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
    pub fn realize(&self, providers: PVec) -> anyhow::Result<RealizedOp> {
        self.op.realize(providers)
    }
}
