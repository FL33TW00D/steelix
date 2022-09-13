use std::{borrow::Cow, sync::Arc};

use num::cast::AsPrimitive;
use onnx::onnx_pb;

use crate::{as_float, BoxOp, DType, DataType, IntoArcTensor, Op, OpGroup, Tensor};

#[derive(Debug, Clone)]
pub struct BatchNormalization {
    pub epsilon: f32,
}

impl BatchNormalization {
    ///Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
    pub fn normalize<T>(
        &self,
        input: &Tensor,
        scale: &Tensor,
        beta: &Tensor,
        mean: &Tensor,
        var: &Tensor,
    ) -> anyhow::Result<Arc<Tensor>>
    where
        T: DataType + num_traits::Float + 'static,
        f32: AsPrimitive<T>,
    {
        let c_dim = input.shape[1];
        let scale = scale
            .to_array_view::<T>()
            .unwrap()
            .into_shape((c_dim,))
            .unwrap();
        let beta = beta
            .to_array_view::<T>()
            .unwrap()
            .into_shape((c_dim,))
            .unwrap();
        let mean = mean
            .to_array_view::<T>()
            .unwrap()
            .into_shape((c_dim,))
            .unwrap();
        let var = var
            .to_array_view::<T>()
            .unwrap()
            .into_shape((c_dim,))
            .unwrap();

        let denominator = var.mapv(|x| (x + self.epsilon.as_()).sqrt());

        let slope = (&scale / &denominator)
            .into_shape((1, c_dim, 1, 1))
            .unwrap();
        let intercept = (beta.to_owned() - (&mean * &scale) / denominator)
            .into_shape((1, c_dim, 1, 1))
            .unwrap();
        let input = input.to_array_view::<T>().unwrap();

        Ok(((slope * input) + intercept).into_arc_tensor())
    }
}

impl Op for BatchNormalization {
    fn name(&self) -> Cow<str> {
        "BatchNormalization".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Normalization
    }

    ///Inputs
    ///X (diff): T
    ///Scale (diff): T1
    ///B: T1
    ///input_mean
    ///input_var
    ///
    ///Outputs
    ///Y: T
    fn realize(&self, providers: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        //TODO: error check
        Ok(vec![as_float!(BatchNormalization::normalize(
            providers[0].dt
        )(
            self,
            &providers[0],
            &providers[1],
            &providers[2],
            &providers[3],
            &providers[4]
        ))?])
    }
}

pub fn build_batchnorm(proto: &onnx_pb::NodeProto) -> Result<BoxOp, anyhow::Error> {
    let epsilon = proto.extract_named_float("epsilon")?.unwrap_or(1e-5);
    Ok(Box::new(BatchNormalization { epsilon }) as BoxOp)
}
