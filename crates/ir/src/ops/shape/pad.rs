use strum_macros::EnumString;

#[derive(EnumString)]
pub enum PaddingMode {
    Constant,
    Reflect,
    Edge,
}

#[derive(Debug, Clone)]
pub struct Pad {
    pads: Vec<usize>,
}

impl Pad {}
