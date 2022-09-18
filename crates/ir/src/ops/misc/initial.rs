use std::{borrow::Cow, sync::Arc};

use crate::{BoxOp, Op, OpGroup, QuadVec, RealizedOp, Tensor};

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

    fn realize(&self, providers: QuadVec) -> anyhow::Result<RealizedOp> {
        if let Some(t) = &self.0 {
            let mut qv = QuadVec::new();
            qv.push(t.clone());
            Ok(RealizedOp::zero_cost(qv))
        } else {
            panic!("Uninitialized input tensor found")
        }
    }

    fn update(&mut self, t: Arc<Tensor>) {
        println!("Setting initial");
        self.set_initial(t);
    }
}

pub fn build_initial() -> Result<BoxOp, anyhow::Error> {
    Ok(Box::new(Initial(None)) as BoxOp)
}
