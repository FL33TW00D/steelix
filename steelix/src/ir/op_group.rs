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
        m.insert(OpGroup::Activation, "lightsalmon");
        m.insert(OpGroup::Constant, "lightgray");
        m.insert(OpGroup::Data, "lightgray");
        m.insert(OpGroup::Layer, "lightsalmon");
        m.insert(OpGroup::Normalization, "lightsalmon");
        m.insert(OpGroup::Transform, "lightsalmon");
        m.insert(OpGroup::Tensor, "lightsalmon");
        m
    };
    pub static ref SHAPE_MAP: HashMap<OpGroup, &'static str> = {
        let mut m = HashMap::new();
        m.insert(OpGroup::Activation, "ellipse");
        m.insert(OpGroup::Constant, "box");
        m.insert(OpGroup::Data, "box");
        m.insert(OpGroup::Layer, "ellipse");
        m.insert(OpGroup::Normalization, "ellipse");
        m.insert(OpGroup::Transform, "ellipse");
        m.insert(OpGroup::Tensor, "ellipse");
        m
    };
}
