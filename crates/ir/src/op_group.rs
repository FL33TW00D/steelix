use lazy_static::lazy_static;
use std::collections::HashMap;

/// OpGroup defines subsets of operations. This is used for colorizing output
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum OpGroup {
    Activation,
    Constant,
    Data,
    Dropout,
    Layer,
    Logic,
    Normalization,
    Pool,
    Shape,
    Tensor,
    Transform,
    Unimplemented,
}

lazy_static! {
    pub static ref COLOUR_MAP: HashMap<OpGroup, &'static str> = {
        let mut m = HashMap::new();
        m.insert(OpGroup::Activation, "salmon");
        m.insert(OpGroup::Constant, "lightgray");
        m.insert(OpGroup::Data, "lightgray");
        m.insert(OpGroup::Layer, "springgreen");
        m
    };
}
