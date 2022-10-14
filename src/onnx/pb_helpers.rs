use std::str::from_utf8;

use crate::onnx_pb::{AttributeProto, NodeProto};
use thiserror::Error;

#[derive(Error, Debug)]
#[error("did not find attribute '{attribute}' for node '{node_name}'")]
pub struct AttributeNotFoundError {
    attribute: String,
    node_name: String,
}

impl NodeProto {
    pub fn get_attribute<T: From<AttributeProto>>(
        &self,
        attribute: &str,
        default: Option<T>,
    ) -> Result<T, AttributeNotFoundError> {
        match (
            self.attribute.iter().find(|attr| attr.name == attribute),
            default,
        ) {
            (Some(attr), _) => Ok(attr.clone().into()),
            (None, Some(default_attr)) => Ok(default_attr),
            (None, None) => Err(AttributeNotFoundError {
                attribute: attribute.to_string(),
                node_name: self.name.to_string(),
            }),
        }
    }
}
impl From<AttributeProto> for Vec<i64> {
    fn from(value: AttributeProto) -> Self {
        value.ints
    }
}

impl From<AttributeProto> for Vec<f32> {
    fn from(value: AttributeProto) -> Self {
        value.floats
    }
}

impl From<AttributeProto> for f32 {
    fn from(value: AttributeProto) -> Self {
        value.f
    }
}

impl From<AttributeProto> for i64 {
    fn from(value: AttributeProto) -> Self {
        value.i
    }
}

impl From<AttributeProto> for String {
    fn from(value: AttributeProto) -> Self {
        from_utf8(&value.s).unwrap().to_string()
    }
}
