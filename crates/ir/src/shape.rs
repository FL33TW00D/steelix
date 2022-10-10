#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shape(pub SmallVec<[usize; 4]>);

use std::{fmt::Display, ops::Deref};

use ndarray::{Dim, IntoDimension, IxDyn, IxDynImpl};
use smallvec::SmallVec;

impl Deref for Shape {
    type Target = SmallVec<[usize; 4]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{}x{}", self[0], self[1], self[2], self[3])
    }
}
