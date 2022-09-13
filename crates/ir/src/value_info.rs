use super::TensorShape;
use onnx::onnx_pb;

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub dimensions: Option<TensorShape>,
}

impl TryFrom<onnx_pb::ValueInfoProto> for ValueInfo {
    type Error = anyhow::Error;

    fn try_from(vip: onnx_pb::ValueInfoProto) -> Result<Self, Self::Error> {
        let name = vip.name;
        if let Some(value_type) = &vip.r#type {
            if let Some(v) = &value_type.value {
                match v {
                    onnx_pb::type_proto::Value::TensorType(t) => {
                        let pb_dims = t.shape.clone().unwrap().dim;

                        let mut dimensions = Vec::with_capacity(pb_dims.len());
                        pb_dims
                            .into_iter()
                            .for_each(|dim| match dim.value.unwrap() {
                                onnx_pb::tensor_shape_proto::dimension::Value::DimValue(v) => {
                                    dimensions.push(v);
                                }
                                onnx_pb::tensor_shape_proto::dimension::Value::DimParam(_) => {
                                    todo!()
                                }
                            });

                        return Ok(Self {
                            name,
                            dimensions: Some(TensorShape { dimensions }),
                        });
                    }
                    onnx_pb::type_proto::Value::SequenceType(_) => todo!(),
                    onnx_pb::type_proto::Value::MapType(_) => todo!(),
                    onnx_pb::type_proto::Value::OptionalType(_) => todo!(),
                    onnx_pb::type_proto::Value::SparseTensorType(_) => todo!(),
                }
            }
        }
        Err(anyhow::anyhow!("BAIL"))
    }
}
