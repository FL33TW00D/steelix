#[derive(Debug, Clone, Default)]
pub struct TensorShape {
    pub dimensions: Vec<i64>,
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for d in self.dimensions.iter().take(self.dimensions.len() - 1) {
            write!(f, "{}x", d).expect("Could not write TensorShape");
        }
        write!(f, "{}", self.dimensions[self.dimensions.len() - 1])
    }
}
