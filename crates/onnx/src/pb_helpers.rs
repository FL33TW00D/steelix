use crate::onnx_pb::{AttributeProto, NodeProto};

impl NodeProto {
    pub fn extract_named_attr(
        &self,
        attr_name: &str,
    ) -> Result<Option<&AttributeProto>, anyhow::Error> {
        let attr = match self.attribute.iter().find(|a| a.name == attr_name) {
            Some(attr) => attr,
            _ => return Ok(None),
        };
        Ok(Some(attr))
    }

    pub fn extract_named_float(&self, attr_name: &str) -> Result<Option<f32>, anyhow::Error> {
        let float = match self.attribute.iter().find(|a| a.name == attr_name) {
            Some(attr) => Some(attr.f),
            _ => return Ok(None),
        };
        Ok(float)
    }

    pub fn extract_named_int(&self, attr_name: &str) -> Result<Option<i64>, anyhow::Error> {
        let int = match self.attribute.iter().find(|a| a.name == attr_name) {
            Some(attr) => Some(attr.i),
            _ => return Ok(None),
        };
        Ok(int)
    }

    pub fn extract_named_intv(&self, attr_name: &str) -> Result<Option<Vec<i64>>, anyhow::Error> {
        let attr = match self.attribute.iter().find(|a| a.name == attr_name) {
            Some(attr) => attr.ints.clone(),
            _ => return Ok(None),
        };
        Ok(Some(attr))
    }
}
