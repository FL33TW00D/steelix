use std::{borrow::Cow, sync::Arc};

use crate::{BoxOp, Op, OpGroup, Tensor};

//Takes an optional tensor which is initialized by the user inputs
#[derive(Debug, Clone)]
pub struct Initial(Option<Arc<Tensor>>);

impl Initial {
    pub fn set_initial(&mut self, t: Arc<Tensor>) {
        self.0 = Some(t);
    }
}

impl Op for Initial {
    fn name(&self) -> Cow<str> {
        "Initial".into()
    }

    fn op_group(&self) -> OpGroup {
        OpGroup::Data
    }

    fn realize(&self, _: Vec<Arc<Tensor>>) -> anyhow::Result<Vec<Arc<Tensor>>> {
        if let Some(t) = &self.0 {
            Ok(vec![Arc::clone(t)])
        } else {
            panic!("Uninitialized input tensor found")
        }
    }

    fn update(&mut self, t: Arc<Tensor>) {
        self.set_initial(t)
    }
}

pub fn build_initial() -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Initial(None)) as BoxOp)
}
