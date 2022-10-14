use crate::{
    ir::{ModelError, Shape},
    shape,
};

use steelix_onnx::onnx_pb;

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub dimensions: Shape,
}

impl TryFrom<onnx_pb::ValueInfoProto> for ValueInfo {
    type Error = ModelError;

    fn try_from(vip: onnx_pb::ValueInfoProto) -> Result<Self, Self::Error> {
        let name = vip.name.clone();
        if let Some(value_type) = &vip.r#type {
            if let Some(v) = &value_type.value {
                match v {
                    onnx_pb::type_proto::Value::TensorType(t) => {
                        let pb_dims = t.shape.clone().unwrap().dim;

                        let mut dimensions = shape!();
                        pb_dims
                            .into_iter()
                            .for_each(|dim| match dim.value.unwrap() {
                                onnx_pb::tensor_shape_proto::dimension::Value::DimValue(v) => {
                                    dimensions.push(v as usize);
                                }
                                onnx_pb::tensor_shape_proto::dimension::Value::DimParam(_) => {
                                    println!("Parameterized dimension found, using 1");
                                    dimensions.push(1); //Pushing 1 for a N batch
                                }
                            });

                        return Ok(Self { name, dimensions });
                    }
                    onnx_pb::type_proto::Value::SequenceType(_) => {
                        return Err(ModelError::UnsupportedType("SequenceType".to_string()));
                    }
                    onnx_pb::type_proto::Value::MapType(_) => {
                        return Err(ModelError::UnsupportedType("MapType".to_string()));
                    }
                    onnx_pb::type_proto::Value::OptionalType(_) => {
                        return Err(ModelError::UnsupportedType("OptionalType".to_string()));
                    }
                    onnx_pb::type_proto::Value::SparseTensorType(_) => {
                        return Err(ModelError::UnsupportedType("SparseTensorType".to_string()));
                    }
                }
            }
        }
        Err(ModelError::UnexpectedError(anyhow::anyhow!(
            "Failed to parse value info proto: {:?}",
            vip,
        )))
    }
}
