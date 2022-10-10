#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shape(pub SmallVec<[usize; 4]>);

use std::{
    fmt::Display,
    ops::{Deref, DerefMut},
};

use smallvec::SmallVec;

impl Deref for Shape {
    type Target = SmallVec<[usize; 4]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Shape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, dim) in self.iter().enumerate() {
            if i > 0 {
                write!(f, "x")?;
            }
            write!(f, "{}", dim)?;
        }
        Ok(())
    }
}
