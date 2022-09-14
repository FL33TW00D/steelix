//here we need to create a hashmap at runtime

use std::collections::HashMap;

use onnx::onnx_pb;

use crate::{
    ops::{activation, binary, nn, pool, shape},
    BoxOp,
};

pub type OpBuilder = fn(node: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error>;
pub type Register = HashMap<String, OpBuilder>;

pub struct OpRegister(Register);

impl Default for OpRegister {
    fn default() -> Self {
        let mut reg = Self(HashMap::new());
        reg.insert("Conv", nn::build_im2col);
        reg.insert("Softmax", activation::build_softmax);
        reg.insert("Clip", activation::build_clip);
        reg.insert("Transpose", shape::build_transpose);
        reg.insert("BatchNormalization", nn::build_batchnorm);
        reg.insert("Add", binary::build_add);
        reg.insert("Squeeze", shape::build_squeeze);
        reg.insert("MatMul", binary::build_matmul);
        reg.insert("AveragePool", pool::build_avgpool);
        reg
    }
}

impl OpRegister {
    pub fn insert(&mut self, s: &'static str, b: OpBuilder) {
        self.0.insert(s.into(), b);
    }

    pub fn get(&self, s: &str) -> Option<&OpBuilder> {
        self.0.get(s)
    }
}