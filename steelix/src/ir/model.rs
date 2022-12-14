use crate::ir::{BoxOp, IntoArcTensor, Op, OpGroup, OpNode, PVec, RealizedOp, Shape, Tensor};
use crate::pvec;

impl<T: Op + ?Sized> Op for Box<T> {
    #[inline]
    fn name(&self) -> std::borrow::Cow<str> {
        (**self).name()
    }

    #[inline]
    fn op_group(&self) -> OpGroup {
        (**self).op_group()
    }

    #[inline]
    fn realize(&self, provider: PVec) -> anyhow::Result<RealizedOp> {
        (**self).realize(provider)
    }
}

use core::fmt::Debug;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
impl Debug for dyn Op {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("{0}")]
    ValidationError(String),
    #[error("{0}")]
    UnsupportedType(String),
    #[error(transparent)]
    UnexpectedError(#[from] anyhow::Error),
}

//                        MODEL GRAPH STRUCTURE
// --------------------------------------------------------------------
// | user inputs | constant initializers | processing nodes | outputs |
// --------------------------------------------------------------------
#[derive(Debug, Default)]
pub struct Model {
    pub nodes: Vec<OpNode<BoxOp>>,
    pub inputs: Vec<usize>,  //IDs of input nodes
    pub outputs: Vec<usize>, //IDs of output nodes
    pub traversal_order: Option<Vec<usize>>,
}

#[derive(Debug, Default)]
pub struct TraversalState {
    pub intermediates: HashMap<usize, PVec>,
}

#[derive(Debug, Default)]
pub struct ModelSummary {
    pub total_flops: usize,
    pub total_params: usize,
    pub op_frequencies: HashMap<String, usize>,
    pub output_shapes: HashMap<usize, Shape>,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_traversal_order(&mut self, order: Vec<usize>) {
        self.traversal_order = Some(order);
    }

    pub fn add_node(&mut self, name: String, op: BoxOp) -> usize {
        let id = self.nodes.len();
        self.nodes.push(OpNode {
            id,
            name,
            op,
            providers: vec![],
            consumers: vec![],
        });
        id
    }

    pub fn add_edge(&mut self, producer_id: usize, consumer_id: usize) {
        let producer = &mut self.nodes[producer_id];
        producer.consumers.push(consumer_id);

        let consumer = &mut self.nodes[consumer_id];
        consumer.providers.push(producer_id);
    }

    ///Performs a DFS from each target node
    pub fn build_traversal_order(mut self) -> Self {
        let mut visited = HashSet::with_capacity(self.nodes.len());
        let mut order: Vec<usize> = vec![];
        for target in self.outputs.clone() {
            if visited.contains(&target) {
                continue;
            }
            let mut current_stack: Vec<(usize, usize)> = vec![(target, 0)]; //node_idx, provider_idx
            while let Some((current_node, current_input)) = current_stack.pop() {
                if self.inputs.contains(&current_node) //reached leaf or all providers visited
                    || current_input == self.nodes[current_node].providers.len()
                {
                    order.push(current_node);
                    visited.insert(current_node);
                } else {
                    let provider = self.nodes[current_node].providers[current_input];

                    if visited.contains(&provider) {
                        //If provider has been added to order, then we know we can move on to
                        //next provider
                        current_stack.push((current_node, current_input + 1));
                    } else {
                        //If we haven't seen the provider, readd the current node
                        //to the stack, and add the provider node to the stack
                        current_stack.push((current_node, current_input));
                        current_stack.push((provider, 0));
                    }
                }
            }
        }
        self.update_traversal_order(order);
        self
    }

    pub fn run(&mut self) -> Result<ModelSummary, ModelError> {
        let mut order = self.traversal_order.clone().unwrap();
        order.pop(); //remove the final node
        let mut traversal_state = TraversalState {
            intermediates: HashMap::new(),
        };

        let mut total_flops = 0;
        let mut total_params = 0;
        let mut output_shapes = HashMap::new();

        let mut op_counts = HashMap::new();
        for node_id in order {
            let node = &mut self.nodes[node_id];

            if node.op.op_group() != OpGroup::Constant {
                *op_counts.entry(node.name.to_owned()).or_insert(0) += 1;
            }

            let providers: PVec = node
                .providers
                .iter()
                .map(|id| {
                    Arc::clone(
                        &traversal_state
                            .intermediates
                            .get(id)
                            .unwrap_or(&pvec![Tensor::default().into_arc_tensor()])[0],
                    )
                })
                .collect();
            let result = node.realize(providers)?;
            total_flops += result.cost.flops;
            total_params += result.cost.parameters;
            output_shapes.insert(node_id, result.outputs[0].shape.clone());

            traversal_state
                .intermediates
                .insert(node_id, result.outputs);
        }
        Ok(ModelSummary {
            total_flops,
            total_params,
            op_frequencies: op_counts,
            output_shapes,
        })
    }
}
